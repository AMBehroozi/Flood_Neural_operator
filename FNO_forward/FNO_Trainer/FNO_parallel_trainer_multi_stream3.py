import os
import sys
import warnings
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F

import random  
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import pandas as pd
from tqdm import tqdm

# Local paths (only if you truly need them)
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

from lib.utilities3 import ensure_directory, check_if_from_ddp, adjust_state_dict
from lib.utiltools import loss_live_plot, AutomaticWeightedLoss

from models.fno3d_encoder import FNO3d
from models.magnifier import MagnifierModel as magnifier

from lib.helper import LargeHydrologyDataset, PaddedIndexProvider, prepare_patch_input, BathtubReconstructor   
from lib.ddp_helpers import setup, cleanup
from lib.helper import coarsen_spatial_tensor
from lib.util import run_nvidia_smi, MHPI


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(rank, world_size,
                config, 
                training_mode,
                model_fn, magnifier_fn, awl_fn, learning_rate,
                scheduler_step, scheduler_gamma, 
                operator_type, T_in, T_out,
                step_time,
                coarsen_mode,
                magnification_factor,
                width_CNO, depth_CNO, kernel_size, unet_depth,  # CNO inputs
                mode1, mode2, mode3, width_FNO,                 # FNO inputs
                wavelet, level, layers, grid_range, width_WNO,  # WNO inputs
                branch_layers, trunk_layers,                    # DeepONet inputs

                epochs, PATH_saved_models, Mode, tag, 
                enable_ig_loss, mask, csv_path, L_x, L_y, criterion,
                save_results, 
                case, batch_size, topo_path, a_path, u_path, train_idx, eval_idx):
    
    setup(rank, world_size)


# ---- Model + DDP ----
    model = model_fn.to(rank)
    magnifier = magnifier_fn.to(rank)

    # Apply DDP to both
    model = DDP(model, device_ids=[rank])
    magnifier = DDP(magnifier, device_ids=[rank])

    param_groups = []

    if training_mode == 'Stage1':   # Only Global Model
        
        print('Stage 1: Pre-train Global Model (FNO) only')
        param_groups.append({'params': model.parameters(), 'lr': learning_rate})
        # param_groups.append({'params': model.parameters()})
        
    elif training_mode == 'Stage2': # Only Magnifier (Freeze Global)
        print('Stage 2: Freeze Global Model, Pre-train Magnifier only')
       
        param_groups.append({'params': magnifier.parameters(), 'lr': learning_rate})
        
    elif training_mode == 'Stage3': # Joint Fine-tuning: Both models
        print('Stage 3: End-to-End Fine-tuning of Global + Magnifier together')
        param_groups.append({'params': model.parameters(), 'lr': learning_rate * 0.1}) 
        param_groups.append({'params': magnifier.parameters(), 'lr': learning_rate})

    else:
        # Exit Scenario: Invalid training_mode provided
        raise ValueError(f"Invalid training_mode: '{training_mode}'. "
                         "Must be 'Stage1', 'Stage2', or 'Stage3'.")

    # ---- Optimizer & Scheduler ----
    optimizer = optim.AdamW(param_groups)
    scheduler = StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


    # ---- Static topo ----
    topo = torch.load(topo_path, map_location='cpu').to(rank)
    GLOBAL_TOPO_MIN = topo.min()
    GLOBAL_TOPO_MAX = topo.max()
    TOPO_RANGE = GLOBAL_TOPO_MAX - GLOBAL_TOPO_MIN + 1e-7
    # ---- Dataset / loaders ----
    full_dataset = LargeHydrologyDataset(a_path, u_path, mask=mask, csv_path=csv_path, Lx=L_x, Ly=L_y)
    train_dataset = Subset(full_dataset, train_idx)
    eval_dataset  = Subset(full_dataset, eval_idx)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    eval_sampler  = DistributedSampler(eval_dataset,  num_replicas=world_size, rank=rank, shuffle=False)

    # drop_last=True avoids uneven last batch across ranks (safer for DDP)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    eval_loader  = DataLoader(eval_dataset,  batch_size=batch_size, sampler=eval_sampler,  drop_last=False)

    # ---- Logs ----
    train_fnolosses, val_losses, val_maglosses  = [], [], []
    train_losses, train_maglosses = [], []

    # ---- Best Model Tracking ----
    best_val_loss = float('inf')
    best_epoch = -1

    outer_loop = tqdm(range(epochs), desc="Progress", position=0)
    torch.cuda.empty_cache()
    
    nx, ny, _ = train_dataset[0][0].shape
    N = 5
    f = magnification_factor
    mx = int(nx/magnification_factor)
    my = int(ny/magnification_factor)
    

    # Set subset_fraction=1.0 to ensure we can find all possible wet patches
    index_provider = PaddedIndexProvider(mx=mx, my=my, N=N, batch_size=32, subset_fraction=1.0)
    
    # --- NEW: Initialize the Physics Reconstructor ---
    reconstructor = BathtubReconstructor(topo, f=f, max_iters=20).to(rank)

    
    WET_BATCH_QUOTA = 8
    accumulation_steps = 2 

    DRY_THRESHOLD = 0.025
    wet_batches_processed = 0


    for ep in outer_loop:
            train_sampler.set_epoch(ep)
            
            # --- 1. SET TRAINING MODES BASED ON STAGE ---
            if training_mode == 'Stage1':
                model.train()
                magnifier.eval()
            elif training_mode == 'Stage2':
                model.eval()     # Freeze global model
                magnifier.train()
            elif training_mode == 'Stage3':
                model.train()    # Joint end-to-end training
                magnifier.train()
            
            total_fnoloss = 0.0
            total_magloss = 0.0
            total_samples = 0
            epoch_total_wet_patches = 0

            # --- 2. TRAINING BATCH LOOP ---
            for batch_data in train_loader:
                batch_data = [item.to(rank) for item in batch_data]
                batch_forcing  = batch_data[0][..., ::step_time]
                
                batch_u0       = batch_data[1][..., ::step_time][..., :T_in]
                batch_u_out_hr = batch_data[1][..., ::step_time][..., T_in:] 
                
                bs = batch_u0.shape[0]
                total_samples += bs
                batch_topo_train = topo.expand(bs, -1, -1)
                # batch_topo_norm = (batch_topo_train - GLOBAL_TOPO_MIN) / TOPO_RANGE
                # --- GLOBAL PASS ---
                optimizer.zero_grad(set_to_none=True)
                U_pred = model(batch_forcing, batch_u0, batch_topo_train)
                # U_pred[U_pred < DRY_THRESHOLD] = 0
                
                batch_u_out_lr = coarsen_spatial_tensor(batch_u_out_hr, N=f, mode=coarsen_mode)
                data_loss = criterion(U_pred, batch_u_out_lr)
                
                # STAGE 1: Standard FNO training
                if training_mode == 'Stage1':
                    data_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_fnoloss += data_loss.item() * bs
                    continue

                # --- NEW: GENERATE GLOBAL BATHTUB BASELINE ---
                # This is the "Identity" for Residual Learning
                with torch.set_grad_enabled(training_mode == 'Stage3'):
                    u_bt_global = reconstructor(U_pred)

                # --- MAGNIFIER PASS (Stage 2 & Stage 3) ---
                # Detach if Stage 2 to save memory; Keep graph for Stage 3
                u_refined_input = U_pred if training_mode == 'Stage3' else U_pred.detach()
                u_bt_input = u_bt_global if training_mode == 'Stage3' else u_bt_global.detach()
                
                # Padding
                u_pad = F.pad(u_refined_input.permute(0, 3, 1, 2), (N, N, N, N), mode='replicate').permute(0, 2, 3, 1)
                u_bt_pad = F.pad(u_bt_input.permute(0, 3, 1, 2), (N*f, N*f, N*f, N*f), mode='replicate').permute(0, 2, 3, 1)
                
                batch_topo_norm = (batch_topo_train - GLOBAL_TOPO_MIN) / TOPO_RANGE
                topo_pad = F.pad(batch_topo_norm, (N*f, N*f, N*f, N*f), mode='replicate')
                target_pad = F.pad(batch_u_out_hr.permute(0, 3, 1, 2), (N*f, N*f, N*f, N*f), mode='replicate').permute(0, 2, 3, 1)
                
                all_batches = list(index_provider.get_batches())
                random.shuffle(all_batches) 

                batch_accumulated_mag_loss = 0.0
                wet_batches_in_step = 0
                wet_batches_processed = 0

                # Patch Processing
                for spatial_batch in all_batches:
                    if wet_batches_processed >= WET_BATCH_QUOTA:
                        break

                    batch_patches_in, batch_patches_bt, batch_patches_trgt = [], [], []
                    for (i_s, j_s) in spatial_batch:
                        p_target = target_pad[:, i_s*f : i_s*f + N*f, j_s*f : j_s*f + N*f, :].unsqueeze(1)
                        if p_target.max() < 0.025: continue 

                        # --- UPDATED: Prepare 3-channel input ---
                        p_in = prepare_patch_input(u_pad, topo_pad, i_s, j_s, N, f, rank, u_bathtub=u_bt_pad)
                        
                        # Store the local bathtub baseline for residual summation
                        p_bt = u_bt_pad[:, i_s*f : i_s*f + N*f, j_s*f : j_s*f + N*f, :].unsqueeze(1)
                        
                        batch_patches_in.append(p_in)
                        batch_patches_bt.append(p_bt)
                        batch_patches_trgt.append(p_target)

                    if not batch_patches_in: continue

                    # Forward through Magnifier to get RESIDUAL
                    # magnifier predicts the CORRECTION only
                    residual = magnifier(torch.cat(batch_patches_in, dim=0))
                    
                    # --- RESIDUAL SUMMATION ---
                    # Final Prediction = Physical Baseline + Neural Correction
                    big_out = torch.cat(batch_patches_bt, dim=0) + residual
                    
                    # Compute loss on the physically augmented output
                    mag_loss_patch = criterion(big_out, torch.cat(batch_patches_trgt, dim=0))
                    
                    # STAGE 2: Individual patch backward (Global is detached)
                    if training_mode == 'Stage2':
                        (mag_loss_patch / accumulation_steps).backward()
                        if (wet_batches_processed + 1) % accumulation_steps == 0:
                            optimizer.step()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                            optimizer.zero_grad(set_to_none=True)
                    
                    # STAGE 3: Aggregate loss for single backward
                    else:
                        batch_accumulated_mag_loss += mag_loss_patch

                    wet_batches_processed += 1
                    wet_batches_in_step += 1
                    num_patches_total = bs * len(batch_patches_in)
                    total_magloss += mag_loss_patch.item() * num_patches_total
                    epoch_total_wet_patches += num_patches_total

                # Final Step for Stage 3 Joint Training
                if training_mode == 'Stage3' and wet_batches_in_step > 0:
                    (batch_accumulated_mag_loss / wet_batches_in_step).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    torch.nn.utils.clip_grad_norm_(magnifier.parameters(), 0.1)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                total_fnoloss += data_loss.item() * bs

            # --- 3. EVALUATION PHASE ---
            model.eval()
            magnifier.eval()
            total_valloss, total_val_magloss = 0.0, 0.0
            total_val_samples, total_val_wet_patches = 0, 0

            # Sample 20% of spatial domain for validation speed
            val_index_provider = PaddedIndexProvider(mx=mx, my=my, N=N, batch_size=8, subset_fraction=0.2)
            val_batches = list(val_index_provider.get_batches())

            with torch.no_grad():
                for batch_data in eval_loader:
                    batch_data = [item.to(rank) for item in batch_data]
                    v_forcing, v_sol = batch_data[0][..., ::step_time], batch_data[1][..., ::step_time]
                    v_u0, v_hr = v_sol[..., :T_in], v_sol[..., T_in:]

                    bs_v = v_u0.shape[0]
                    total_val_samples += bs_v
                    v_topo = topo.expand(bs_v, -1, -1)

                    # Global Validation
                    v_coarse_pred = model(v_forcing, v_u0, v_topo)
                    v_coarse_pred[v_coarse_pred < DRY_THRESHOLD] = 0
                    total_valloss += criterion(v_coarse_pred, coarsen_spatial_tensor(v_hr, N=f, mode=coarsen_mode)).item() * bs_v

                    if training_mode == 'Stage1': continue

                    # --- NEW: Generate Global Bathtub for Validation ---
                    v_u_bt_global = reconstructor(v_coarse_pred)

                    # Magnifier Validation Prep
                    v_u_pad = F.pad(v_coarse_pred.permute(0, 3, 1, 2), (N,N,N,N), mode='replicate').permute(0,2,3,1)
                    v_u_bt_pad = F.pad(v_u_bt_global.permute(0, 3, 1, 2), (N*f,N*f,N*f,N*f), mode='replicate').permute(0,2,3,1)
                    v_topo_pad = F.pad((v_topo - GLOBAL_TOPO_MIN)/TOPO_RANGE, (N*f,N*f,N*f,N*f), mode='replicate')
                    v_trgt_pad = F.pad(v_hr.permute(0,3,1,2), (N*f,N*f,N*f,N*f), mode='replicate').permute(0,2,3,1)

                    for spatial_batch in val_batches:
                        v_p_in, v_p_bt, v_p_trgt = [], [], []
                        for (i_s, j_s) in spatial_batch:
                            p_t = v_trgt_pad[:, i_s*f : i_s*f + N*f, j_s*f : j_s*f + N*f, :].unsqueeze(1)
                            if p_t.max() < DRY_THRESHOLD: continue
                            
                            # --- UPDATED: Pass bathtub to prepare_patch_input ---
                            v_p_in.append(prepare_patch_input(v_u_pad, v_topo_pad, i_s, j_s, N, f, rank, u_bathtub=v_u_bt_pad))
                            
                            # Capture local bathtub slice for residual sum
                            p_bt_local = v_u_bt_pad[:, i_s*f : i_s*f + N*f, j_s*f : j_s*f + N*f, :].unsqueeze(1)
                            v_p_bt.append(p_bt_local)
                            v_p_trgt.append(p_t)

                        if v_p_in:
                            # Forward pass (Residual prediction)
                            v_residual = magnifier(torch.cat(v_p_in, dim=0))
                            
                            # --- RESIDUAL SUMMATION ---
                            v_final_out = torch.cat(v_p_bt, dim=0) + v_residual
                            
                            num_w_total = bs_v * len(v_p_in)
                            total_val_magloss += criterion(v_final_out, torch.cat(v_p_trgt, dim=0)).item() * num_w_total
                            total_val_wet_patches += num_w_total

            # --- 4. EPOCH LOGGING & NORMALIZATION ---
            epoch_fnoloss = total_fnoloss / total_samples
            epoch_valloss = total_valloss / total_val_samples
            
            # Ensure lists are updated even in Stage 1 to prevent length mismatch errors
            epoch_magloss = total_magloss / max(epoch_total_wet_patches, 1) if training_mode != 'Stage1' else 0.0
            epoch_val_magloss = total_val_magloss / max(total_val_wet_patches, 1) if training_mode != 'Stage1' else 0.0
            
            train_losses.append(epoch_fnoloss)
            train_maglosses.append(epoch_magloss)
            val_losses.append(epoch_valloss)
            val_maglosses.append(epoch_val_magloss)

            # --- 5. LOGGING AND CHECKPOINTING (UPDATED) ---
            if rank == 0:
                df_main = pd.DataFrame({'Train FNO Loss': train_losses, 'Val FNO Loss': val_losses})
                df_magnifier = pd.DataFrame({'Train Mag Loss': train_maglosses, 'Val Mag Loss': val_maglosses})

                if save_results and (ep % 5 == 0):
                    # Current epoch checkpoint
                    checkpoint_data = {
                        'config_arg': config,
                        'training_mode': training_mode,
                        'config': {
                            'Nx': nx, 'Ny': ny, 'T_in': T_in, 'T_out': T_out,
                            'N_window': N, 'f_upscale': f, 'dry_threshold': DRY_THRESHOLD,
                            'Mode1_G':mode1, 'Mode2_G': mode2, 'Mode3_G': mode3, 'width_FNO': width_FNO,
                            'training_stage': training_mode
                        },
                        'epoch': ep,
                        'model_state_dict': model.state_dict(),
                        'magnifier_state_dict': magnifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_df_main': df_main,
                        'loss_df_magnifier': df_magnifier,
                    }
                    torch.save(checkpoint_data, f"{PATH_saved_models}/saved_model_{Mode}.pth")
                    print(f"\n[Checkpoint] Saved Epoch {ep+1} for {training_mode}")

                if training_mode == "Stage1":
                    epoch_valloss_criteria = epoch_valloss
                else:
                    epoch_valloss_criteria = epoch_val_magloss

                # Check if this is the best model so far
                if epoch_valloss_criteria < best_val_loss:
                    best_val_loss = epoch_valloss_criteria
                    best_epoch = ep
                    
                    # Best model checkpoint
                    best_checkpoint_data = {
                        'config_arg': config,
                        'training_mode': training_mode,
                        'config': {
                            'Nx': nx, 'Ny': ny, 'T_in': T_in, 'T_out': T_out,
                            'N_window': N, 'f_upscale': f, 'dry_threshold': DRY_THRESHOLD,
                            'Mode1_G':mode1, 'Mode2_G': mode2, 'Mode3_G': mode3, 'width_FNO': width_FNO,
                            'training_stage': training_mode
                        },
                        'epoch': ep,
                        'best_val_loss': best_val_loss,
                        'model_state_dict': model.state_dict(),
                        'magnifier_state_dict': magnifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_df_main': df_main,
                        'loss_df_magnifier': df_magnifier,
                    }
                    torch.save(best_checkpoint_data, f"{PATH_saved_models}/saved_model_{Mode}_best.pth")
                    print(f"\n[Best Model] Saved at Epoch {ep+1} with Val Loss: {best_val_loss:.2e}")

                outer_loop.set_description(f"Epoch {ep + 1}/{epochs} [{training_mode}]")
                outer_loop.set_postfix(fno_train=f'{epoch_fnoloss:.2e}', fno_val=f'{epoch_valloss:.2e}', 
                                       mag_train=f'{epoch_magloss:.2e}', mag_val=f'{epoch_val_magloss:.2e}',
                                       best_ep=best_epoch+1)

            scheduler.step()

    # --- 6. FINAL BEST MODEL SUMMARY ---
    if rank == 0 and save_results:
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Model: Epoch {best_epoch+1} with Val Loss: {best_val_loss:.2e}")
        print(f"Best model saved at: {PATH_saved_models}/best_model_{Mode}.pth")
        print(f"Latest model saved at: {PATH_saved_models}/saved_model_{Mode}.pth")
        print(f"{'='*60}\n")

    cleanup()


# %%
def main(
    config_arg,
    enable_ig_loss,
    save_results,
    global_checkpoint_path,
    mag_checkpoint_path,
    training_mode,
    topo_path, a_path, u_path,
    train_idx, eval_idx,
    case, tag,
    batch_size, epochs, learning_rate,
    scheduler_step, scheduler_gamma,
    operator_type, T_in, T_out, 
    step_time,
    coarsen_mode,
    nx, ny,
    magnification_factor,
    mask=None, csv_path=None, L_x=None, L_y=None,
    width_CNO=None, depth_CNO=None, kernel_size=None, unet_depth=None,
    mode1=None, mode2=None, mode3=None, width_FNO=None,
    wavelet=None, level=None, layers=None, grid_range=None, width_WNO=None,
    branch_layers=None, trunk_layers=None
):
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices found. DDP requires GPUs for nccl backend.")

    print(f"Number of GPUs: {world_size}")
    device = torch.device("cuda")
    print(f"Using device: {device}")


    # Build experiment name (train_size inferred from indices)
    train_size = len(train_idx)
    Mode = (
        f"{case}_{'IG_Enable' if enable_ig_loss else 'IG_Disable'}"
        f"_Nx_{nx}_Ny_{ny}_Tin_{T_in}_Tout_{T_out}"
        f"_Samp_{tag}_{operator_type}"
        f"_DDP_{train_size}"
    )
    print(f"Mode: {Mode}")

    # Paths
    main_path = os.path.join("experiments", case)
    PATH_saved_models = os.path.join(main_path, "saved_models")
    ensure_directory(PATH_saved_models)

    if global_checkpoint_path != None:
        print('Loading pre-trained global model ...')
        checkpoint = torch.load(global_checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract configuration from checkpoint
        config = checkpoint['config']
        m1, m2, m3 = config['Mode1_G'], config['Mode2_G'], config['Mode3_G']
        width_val = config['width_FNO']
        
        # Initialize and load weights
        model_fn = FNO3d(
            T_in=T_in, T_out=T_out,
            modes_x=m1, modes_y=m2, modes_t=m3,
            width=width_val,
            encoder_kernel_size_x=82,
            encoder_kernel_size_y=41,
            encoder_num_layers=4
        )

        state_dict = checkpoint['model_state_dict']
        if check_if_from_ddp(checkpoint):   
            state_dict = adjust_state_dict(state_dict, model_fn)
        model_fn.load_state_dict(state_dict)

    else:
        print('Creating a raw global model ...')
        # Initialize with default/global variables if no checkpoint is used
        model_fn = FNO3d(
            T_in=T_in, T_out=T_out,
            modes_x=mode1, # Uses variables defined elsewhere in your script
            modes_y=mode2, 
            modes_t=mode3,
            width=width_FNO,
            encoder_kernel_size_x=82,
            encoder_kernel_size_y=41,
            encoder_num_layers=4
        )
        
    model_fn = model_fn.to(device)


    if mag_checkpoint_path != None:
        print('Loading pre-trained magnifier model ...')
        # Path to your specific magnifier checkpoint
        checkpoint_mag = torch.load(mag_checkpoint_path, map_location='cpu', weights_only=False)
        
        # Initialize magnifier architecture
        magnifier_fn = magnifier(
        in_channels=3,
        base_channels=20,
        num_fno_blocks=2,
        fno_modes_x=8,
        fno_modes_y=8,
        num_refinement_blocks=3,
        num_residual_per_block=3,
        channel_multipliers=[1.0,  2.0, 2.0],
        dropout=0.1,
        use_attention=False,
        use_pyramid_pooling=False,
        use_gradient_checkpointing=False
        )
        
        # Extract state dict (using the specific key for magnifier weights)
        state_dict_mag = checkpoint_mag['magnifier_state_dict']
        
        # Handle DDP state dict adjustment
        if check_if_from_ddp(checkpoint_mag):
            state_dict_mag = adjust_state_dict(state_dict_mag, magnifier_fn)
            
        magnifier_fn.load_state_dict(state_dict_mag)

    else:
        print('Creating a raw magnifier model ...')
        # Fresh initialization for the magnifier
        magnifier_fn = magnifier(
        in_channels=3,
        base_channels=20,
        num_fno_blocks=2,
        fno_modes_x=8,
        fno_modes_y=8,
        num_refinement_blocks=3,
        num_residual_per_block=3,
        channel_multipliers=[1.0,  2.0, 2.0],
        dropout=0.1,
        use_attention=False,
        use_pyramid_pooling=False,
        use_gradient_checkpointing=False
        )

    magnifier_fn = magnifier_fn.to(device)


    # Only needed if IG is enabled (still safe to create)
    ss = 2 if enable_ig_loss else 1
    awl_fn = AutomaticWeightedLoss(ss)
    criterion = nn.MSELoss()

    torch.multiprocessing.spawn(
        train_model,
        args=(
            world_size,
            config_arg, 
            training_mode, 
            model_fn, magnifier_fn, awl_fn, 
            learning_rate,
            scheduler_step, scheduler_gamma,
            operator_type, T_in, T_out,
            step_time,
            coarsen_mode,
            magnification_factor,
            width_CNO, depth_CNO, kernel_size, unet_depth,
            mode1, mode2, mode3, width_FNO,
            wavelet, level, layers, grid_range, width_WNO,
            branch_layers, trunk_layers,

            epochs, PATH_saved_models, Mode, tag,
            enable_ig_loss, mask, csv_path, L_x, L_y, criterion,
            save_results,
            case, batch_size, topo_path, a_path, u_path, train_idx, eval_idx
        ),
        nprocs=world_size,
        join=True
    )

# %%

if __name__ == "__main__":
    
    # 1. Initialization & Config Loading
    
    # Dam break
    # cfg = load_config('FNO_forward/FNO_Trainer/configs/dam_break_config_stage1.yml')    
    cfg = load_config('FNO_forward/FNO_Trainer/configs/dam_break_config_stage2.yml')    
    
    # flooding
    # cfg = load_config('FNO_forward/FNO_Trainer/configs/flooding_config_stage1.yml')
    # cfg = load_config('FNO_forward/FNO_Trainer/configs/flooding_config_stage2.yml')
    
    run_nvidia_smi()
    MHPI()
    warnings.filterwarnings("ignore", message="incompatible copy of pydevd already imported")

    # 2. Extract Paths & IO
    paths = cfg['paths']
    topo_path = os.path.join(paths['main_path'], paths['topo_file'])
    a_path    = os.path.join(paths['main_path'], paths['input_file'])
    u_path    = os.path.join(paths['main_path'], paths['solution_file'])
    
    mag_checkpoint_path    = paths.get('mag_checkpoint')
    global_checkpoint_path = paths.get('global_checkpoint')

    # 3. Experiment Logic
    exp = cfg['experiment']
    case, tag = exp['case'], exp['tag']
    training_mode = exp['training_mode']
    enable_ig_loss, save_results = exp['enable_ig_loss'], exp['save_results']

    # 4. Dataset Initialization & Deterministic Splitting
    tmp_ds = LargeHydrologyDataset(a_path, u_path)
    g = torch.Generator().manual_seed(exp['seed'])
    perm = torch.randperm(len(tmp_ds), generator=g).tolist()
    
    train_idx = perm[:exp['train_size']]
    eval_idx  = perm[exp['train_size'] : exp['train_size'] + exp['eval_size']]

    # 5. Training Hyperparameters
    train_cfg = cfg['training']
    batch_size = train_cfg['batch_size']
    epochs = train_cfg['epochs']
    learning_rate = train_cfg['learning_rate']
    scheduler_step = train_cfg['scheduler_step']
    scheduler_gamma = train_cfg['scheduler_gamma']

    data_mask_cfg = cfg['data_mask']
    mask = data_mask_cfg['mask']
    csv_path = data_mask_cfg['csv_path']
    L_x = data_mask_cfg['L_x']
    L_y = data_mask_cfg['L_y']

    # 6. Model Parameters
    m_cfg = cfg['model']
    operator_type = m_cfg['operator_type']
    nx, ny = m_cfg['nx'], m_cfg['ny']
    T_in = m_cfg['t_in']
    T_out = m_cfg['total_steps'] - T_in
    step_time = m_cfg['step_time']
    coarsen_mode =  m_cfg['coarsen_mode']
    magnification_factor = m_cfg['magnification_factor']

    # 7. Model-Specific Logic (e.g., FNO)
    fno_cfg =   cfg['fno']
    mode_x =    cfg['fno']['mode_x']
    mode_y =    cfg['fno']['mode_y']
    mode_t =    cfg['fno']['mode_t']
    width_FNO = cfg['fno']['width']

    print(f"--- Config Loaded for {case} ({training_mode}) ---")
    print(f"Training on {len(train_idx)} samples; Evaluating on {len(eval_idx)} samples.")

    # CNO inputs
    width_CNO, depth_CNO = 128, 4
    kernel_size = 3
    unet_depth = 4

    # WNO inputs
    wavelet = 'db6'  # Wavelet basis function
    level = 4        # Level of wavelet decomposition
    width_WNO = 30   # Uplifting dimension
    layers = 4       # Number of wavelet layers
    grid_range = [1, 1, 1]

    # DeepONet inputs

    branch_layers = [64, 128, 128, 128, 64]
    trunk_layers =  [64, 128, 128, 128, 64]

    # Call the main function with organized inputs
    main(
        cfg,
        enable_ig_loss, save_results, 
        global_checkpoint_path, 
        mag_checkpoint_path,
        training_mode,
        topo_path, a_path, u_path, train_idx, eval_idx, case, 
        tag, batch_size, 
        epochs, learning_rate, scheduler_step, scheduler_gamma, 
        operator_type, T_in, T_out, 
        step_time,
        coarsen_mode,
        nx=nx, ny=ny,
        magnification_factor=magnification_factor,
        mask=mask, csv_path=csv_path, L_x=L_x, L_y=L_y,
        # CNO inputs
        width_CNO=width_CNO if operator_type == 'CNO' else None,
        depth_CNO=depth_CNO if operator_type == 'CNO' else None,
        kernel_size=kernel_size if operator_type == 'CNO' else None,
        unet_depth=unet_depth if operator_type == 'CNO' else None,
        # FNO inputs
        mode1=mode_x if operator_type == 'FNO' else None,
        mode2=mode_y if operator_type == 'FNO' else None,
        mode3=mode_t if operator_type == 'FNO' else None,
        width_FNO=width_FNO if operator_type == 'FNO' else None,
        # WNO inputs
        wavelet=wavelet if operator_type == 'WNO' else None,
        level=level if operator_type == 'WNO' else None,
        layers=layers if operator_type == 'WNO' else None,
        grid_range=grid_range if operator_type == 'WNO' else None,
        width_WNO=width_WNO if operator_type == 'WNO' else None,
        # DeepONet inputs
        branch_layers=branch_layers if operator_type == 'DeepONet' else None,
        trunk_layers=trunk_layers if operator_type == 'DeepONet' else None
    )

