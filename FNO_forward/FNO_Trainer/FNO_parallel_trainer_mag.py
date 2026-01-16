import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F

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

from lib.utilities3 import ensure_directory
from lib.utiltools import loss_live_plot, AutomaticWeightedLoss

from models.fno3d_encoder import FNO3d
from models.magnifier import MagnifierModel as magnifier

from lib.helper import LargeHydrologyDataset
from lib.ddp_helpers import setup, cleanup
from lib.helper import coarsen_spatial_tensor
from lib.helper import PaddedIndexProvider, prepare_patch_input


def train_model(rank, world_size, model_fn, magnifier_fn, awl_fn, learning_rate, 
                operator_type, T_in, T_out,
                magnification_factor,
                width_CNO, depth_CNO, kernel_size, unet_depth,  # CNO inputs
                mode1, mode2, mode3, width_FNO,                 # FNO inputs
                wavelet, level, layers, grid_range, width_WNO,  # WNO inputs
                branch_layers, trunk_layers,                    # DeepONet inputs

                epochs, PATH_saved_models, Mode, num_samples_x_y, 
                enable_ig_loss, L_x, L_y, criterion, plot_live_loss, 
                case, batch_size, topo_path, a_path, u_path, train_idx, eval_idx):
    
    setup(rank, world_size)

    # ---- Model + DDP ----
    model = model_fn.to(rank)  # ideally model_fn is a fresh instance per rank
    magnifier = magnifier_fn.to(rank)
   
    model = DDP(model, device_ids=[rank])
    magnifier = DDP(magnifier, device_ids=[rank])
 
    # ---- Optimizer (AWL only if IG enabled) ----
    if enable_ig_loss:
        awl = awl_fn.to(rank)
        awl = DDP(awl, device_ids=[rank])
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'lr': learning_rate},
             {'params': magnifier.parameters(), 'lr': learning_rate},
             {'params': awl.parameters(),   'lr': learning_rate}]
        )
    else:
        awl = None
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'lr': learning_rate},
             {'params': magnifier.parameters(), 'lr': learning_rate}]
        )

    scheduler = StepLR(optimizer, step_size=50, gamma=0.85)

    # ---- Static topo ----
    topo = torch.load(topo_path, map_location='cpu').to(rank)

    # ---- Dataset / loaders ----
    full_dataset = LargeHydrologyDataset(a_path, u_path)
    train_dataset = Subset(full_dataset, train_idx)
    eval_dataset  = Subset(full_dataset, eval_idx)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    eval_sampler  = DistributedSampler(eval_dataset,  num_replicas=world_size, rank=rank, shuffle=False)

    # drop_last=True avoids uneven last batch across ranks (safer for DDP)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    eval_loader  = DataLoader(eval_dataset,  batch_size=batch_size, sampler=eval_sampler,  drop_last=False)

    # ---- Logs ----
    train_fnolosses, train_iglosses, val_losses = [], [], []
    outer_loop = tqdm(range(epochs), desc="Progress", position=0)
    torch.cuda.empty_cache()
    
    nx, ny, _ = train_dataset[0][0].shape
    N = 5
    f = magnification_factor
    mx = int(nx/magnification_factor)
    my = int(ny/magnification_factor)
    
    index_provider = PaddedIndexProvider(mx=mx, my=my, N=N, batch_size=8, subset_fraction=0.2)

    for ep in outer_loop:
        train_sampler.set_epoch(ep)
        
        model.train()      # Model 1
        magnifier.train()  # Model 2
        
        total_fnoloss = 0.0
        total_igloss  = 0.0
        total_magloss = 0.0
        total_samples = 0

        for batch_data in train_loader:
            batch_data = [item.to(rank) for item in batch_data]
            
            # [nb, nx, ny, nt]
            batch_forcing  = batch_data[0]
            batch_u0       = batch_data[1][..., :T_in]
            batch_u_out_hr = batch_data[1][..., T_in:] 
            
            bs = batch_u0.shape[0]
            total_samples += bs
            batch_topo = topo.expand(bs, -1, -1) # [bs, nx, ny]

            # ---------------------------------------------------------
            # 1. GLOBAL COARSE PASS (MODEL 1)
            # ---------------------------------------------------------
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass: [bs, mx, my, T_out]
            U_pred = model(batch_forcing, batch_u0, batch_topo)
            
            # Coarsen Ground Truth for Loss
            batch_u_out_lr = coarsen_spatial_tensor(batch_u_out_hr, N=f, mode='bilinear')
            data_loss = criterion(U_pred, batch_u_out_lr)
            
            if enable_ig_loss:
                # ... IG logic stays here ...
                loss1 = awl(data_loss, ig_loss)
            else:
                ig_loss = data_loss.new_tensor(0.0)
                loss1 = data_loss

            loss1.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() # Update Model 1 weights

            # ---------------------------------------------------------
            # 2. PATCH REFINEMENT PASS (MAGNIFIER)
            # ---------------------------------------------------------
            # Use .detach() to ensure Model 1 gradients don't accumulate in the spatial loop
            u_pad = F.pad(U_pred.detach().permute(0, 3, 1, 2), (N, N, N, N), mode='replicate').permute(0, 2, 3, 1)
            topo_pad = F.pad(batch_topo, (N*f, N*f, N*f, N*f), mode='replicate')
            target_pad = F.pad(batch_u_out_hr.permute(0, 3, 1, 2), (N*f, N*f, N*f, N*f), mode='replicate').permute(0, 2, 3, 1)

            if Gradient_Accumulation:
                accumulation_steps = 10 
                optimizer.zero_grad(set_to_none=True) 

                # spatial_batch is a subset based on index_provider.subset_fraction
                for i, spatial_batch in enumerate(index_provider.get_batches()):
                    batch_patches_in = []
                    batch_patches_trgt = []

                    for (i_s, j_s) in spatial_batch:
                        # [bs, 2, Pf, Pf, T_out]
                        p_in = prepare_patch_input(u_pad, topo_pad, i_s, j_s, N, f, rank)
                        # [bs, 1, Pf, Pf, T_out]
                        p_target = target_pad[:, i_s*f : i_s*f + N*f, j_s*f : j_s*f + N*f, :].unsqueeze(1)              
                        
                        batch_patches_in.append(p_in)
                        batch_patches_trgt.append(p_target)

                    # [bs * nb_i, 2, Pf, Pf, T_out]
                    big_in = torch.cat(batch_patches_in, dim=0)
                    big_trgt = torch.cat(batch_patches_trgt, dim=0)

                    big_out = magnifier(big_in)
                    
                    # Mean loss over virtual batch
                    mag_loss_batch = criterion(big_out, big_trgt) / accumulation_steps
                    mag_loss_batch.backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    
                    total_magloss += (mag_loss_batch.item() * accumulation_steps) * (bs * len(spatial_batch))

                # Final step for leftovers
                if (i + 1) % accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            else:
                # Standard Step-by-Step Refinement
                for spatial_batch in index_provider.get_batches():
                    optimizer.zero_grad(set_to_none=True)
                    
                    batch_patches_in = []
                    batch_patches_trgt = []

                    for (i_s, j_s) in spatial_batch:
                        p_in = prepare_patch_input(u_pad, topo_pad, i_s, j_s, N, f, rank)
                        p_target = target_pad[:, i_s*f : i_s*f + N*f, j_s*f : j_s*f + N*f, :].unsqueeze(1)              
                        batch_patches_in.append(p_in)
                        batch_patches_trgt.append(p_target)

                    big_in = torch.cat(batch_patches_in, dim=0)
                    big_trgt = torch.cat(batch_patches_trgt, dim=0)

                    big_out = magnifier(big_in)
                    mag_loss = criterion(big_out, big_trgt)
                    mag_loss.backward()
                    optimizer.step()
                    
                    total_magloss += mag_loss.item() * (bs * len(spatial_batch))

            # Accumulate metrics
            total_fnoloss += data_loss.item() * bs
            total_igloss  += ig_loss.item() * bs

        # Final Epoch Statistics
        # Note: Use num_subset because that is how many windows we actually processed
        epoch_fnoloss = total_fnoloss / total_samples
        epoch_magloss = total_magloss / (total_samples * index_provider.num_subset)
        
        # ---------------- Evaluation phase ----------------
        model.eval()
        magnifier.eval()
        
        total_valloss = 0.0
        total_val_magloss = 0.0
        total_val_samples = 0

        # We use a non-randomized, full systematic sweep for validation 
        # to get a stable, reproducible score (subset_fraction=1.0)
        val_index_provider = PaddedIndexProvider(mx=mx, my=my, N=N, batch_size=8, subset_fraction=1.0)

        with torch.no_grad():
            for batch_data in eval_loader:
                batch_data = [item.to(rank) for item in batch_data]

                # batch_forcing: [bs, nx, ny, nt]
                # batch_u_out_hr: [bs, nx, ny, T_out]
                batch_forcing  = batch_data[0]
                batch_u0       = batch_data[1][..., :T_in]
                batch_u_out_hr = batch_data[1][..., T_in:]

                bs = batch_u0.shape[0]
                total_val_samples += bs
                batch_topo = topo.expand(bs, -1, -1)

                # 1. ─── Model 1 Validation (Global Coarse) ───
                # U_pred: [bs, mx, my, T_out]
                U_pred = model(batch_forcing, batch_u0, batch_topo)
                
                # Coarsen ground truth to match global model resolution
                batch_u_out_lr = coarsen_spatial_tensor(batch_u_out_hr, N=f, mode='bilinear')
                val_loss = criterion(U_pred, batch_u_out_lr)
                total_valloss += val_loss.item() * bs

                # 2. ─── Magnifier Validation (Local Refinement) ───
                # Prepare padded tensors for full-domain sweep
                u_pad = F.pad(U_pred.permute(0, 3, 1, 2), (N, N, N, N), mode='replicate').permute(0, 2, 3, 1)
                topo_pad = F.pad(batch_topo, (N*f, N*f, N*f, N*f), mode='replicate')
                target_pad = F.pad(batch_u_out_hr.permute(0, 3, 1, 2), (N*f, N*f, N*f, N*f), mode='replicate').permute(0, 2, 3, 1)

                # Loop through ALL windows in the domain for a complete high-res assessment
                for spatial_batch in val_index_provider.get_batches():
                    batch_patches_in = []
                    batch_patches_trgt = []

                    for (i_s, j_s) in spatial_batch:
                        # Extract and prepare input: [bs, 2, Pf, Pf, T_out]
                        p_in = prepare_patch_input(u_pad, topo_pad, i_s, j_s, N, f, rank)
                        # Extract target patch: [bs, 1, Pf, Pf, T_out]
                        p_target = target_pad[:, i_s*f : i_s*f + N*f, j_s*f : j_s*f + N*f, :].unsqueeze(1)
                        
                        batch_patches_in.append(p_in)
                        batch_patches_trgt.append(p_target)

                    # Flatten batch: [bs * batch_size, ...]
                    big_in = torch.cat(batch_patches_in, dim=0)
                    big_trgt = torch.cat(batch_patches_trgt, dim=0)

                    # Magnifier prediction
                    big_out = magnifier(big_in)
                    mag_val_loss = criterion(big_out, big_trgt)
                    
                    # Accumulate loss based on number of patches in this spatial batch
                    total_val_magloss += mag_val_loss.item() * (bs * len(spatial_batch))

        # 3. ─── Metrics Normalization ───
        epoch_valloss = total_valloss / total_val_samples
        
        # total_val_magloss is divided by (total_samples * total_spatial_windows)
        epoch_val_magloss = total_val_magloss / (total_val_samples * val_index_provider.num_total_windows)
        
        val_losses.append(epoch_valloss)
        val_maglosses.append(epoch_val_magloss)

        # ---------------- Logging / saving ----------------
        losses_dict = {
            'Train FNO Loss': train_fnolosses,
            'Train Mag Loss': train_maglosses,
            'Val FNO Loss': val_losses,
            'Val Mag Loss': val_maglosses
        }
        if enable_ig_loss:
            losses_dict['Train IG loss'] = train_iglosses

        # Convert to DataFrame for persistence
        df = pd.DataFrame(losses_dict)

        # Ensure we only save on the primary rank in DDP
        if save_results and (ep % 5 == 0) and (rank == 0):
            torch.save({
                'config': {
                    # --- Original Model 1 Config ---
                    'operator_type': operator_type,
                    'enable_ig_loss': enable_ig_loss,
                    'Nx': nx,
                    'Ny': ny,
                    'T_in': T_in,
                    'T_out': T_out,
                    'width_CNO': width_CNO,
                    'depth_CNO': depth_CNO,
                    'kernel_size': kernel_size,
                    'unet_depth': unet_depth,
                    'mode1': mode1,
                    'mode2': mode2,
                    'mode3': mode3,
                    'width_FNO': width_FNO,
                    'wavelet': wavelet,
                    'level': level,
                    'layers': layers,
                    'grid_range': grid_range,
                    'width_WNO': width_WNO,
                    'branch_layers': branch_layers,
                    'trunk_layers': trunk_layers,
                    
                    # --- New Magnifier / Refinement Config ---
                    'mx_coarse': mx,
                    'my_coarse': my,
                    'N_window': N,           # Coarse window size (e.g., 5)
                    'f_upscale': f,          # Magnification factor (e.g., 10)
                    'Pf_fine': N * f,        # Resulting fine patch size
                    'subset_fraction': 0.2,  # Plan 1: Stochastic subsampling %
                    'accumulation_steps': 10 if Gradient_Accumulation else 1 # Plan 2
                },
                'epoch': ep,
                'model_state_dict': model.state_dict(),          # Global Model
                'magnifier_state_dict': magnifier.state_dict(),  # Magnifier Model
                'optimizer_state_dict': optimizer.state_dict(),  # Shared Optimizer
                'loss_df': df,                                   # Full loss history
            }, PATH_saved_models + f'/saved_model_{Mode}.pth')
            
            print(f"--- Checkpoint saved at Epoch {ep+1} ---")


        # Live Plotting
        if plot_live_loss and (rank == 0):
            loss_live_plot(losses_dict)

        scheduler.step()

        # Update Progress Bar with current results
        outer_loop.set_description(f"Epoch {ep + 1}/{epochs}")
        outer_loop.set_postfix(
            fno_v=f'{epoch_valloss:.2e}',
            mag_v=f'{epoch_val_magloss:.2e}',
            mag_t=f'{epoch_magloss:.2e}' # From training phase
        )





        # # ---------------- Evaluation phase ----------------
        # model.eval()
        # total_valloss = 0.0
        # total_val_samples = 0

        # with torch.no_grad():
        #     for batch_data in eval_loader:
        #         batch_data = [item.to(rank) for item in batch_data]

        #         batch_forcing = batch_data[0]
        #         batch_u0      = batch_data[1][..., :T_in]
        #         batch_u_out   = batch_data[1][..., T_in:]

        #         bs = batch_u0.shape[0]
        #         total_val_samples += bs

        #         batch_topo = topo.expand(bs, -1, -1)

        #         U_pred = model(batch_forcing, batch_u0, batch_topo)
        #         val_loss = criterion(U_pred, batch_u_out)

        #         total_valloss += val_loss.item() * bs

        # epoch_valloss = total_valloss / total_val_samples
        # val_losses.append(epoch_valloss)

        # # ---------------- Logging / saving ----------------
        # losses_dict = {
        #     'Training FNO Loss': train_fnolosses,
        #     'Validation Loss': val_losses
        # }
        # if enable_ig_loss:
        #     losses_dict['Train IG loss'] = train_iglosses

        # df = pd.DataFrame(losses_dict)

        # save_results = False
        # if save_results and (ep % 5 == 0) and (rank == 0):
        #     torch.save({
        #         'config': {
        #             'operator_type': operator_type,
        #             'enable_ig_loss': enable_ig_loss,
        #             'Nx': nx,
        #             'Ny': ny,
        #             'T_in': T_in,
        #             'T_out': T_out,
        #             'width_CNO': width_CNO,
        #             'depth_CNO': depth_CNO,
        #             'kernel_size': kernel_size,
        #             'unet_depth': unet_depth,
        #             'mode1': mode1,
        #             'mode2': mode2,
        #             'mode3': mode3,
        #             'width_FNO': width_FNO,
        #             'wavelet': wavelet,
        #             'level': level,
        #             'layers': layers,
        #             'grid_range': grid_range,
        #             'width_WNO': width_WNO,
        #             'branch_layers': branch_layers,
        #             'trunk_layers': trunk_layers,
        #         },
        #         'epoch': ep,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss_df': df,          # clearer name
        #     }, PATH_saved_models + f'/saved_model_{Mode}.pth')

        # if plot_live_loss and (ep % 1 == 0):
        #     loss_live_plot(losses_dict)

        # scheduler.step()

        # outer_loop.set_description(f"Progress (Epoch {ep + 1}/{epochs}) Mode: {Mode}")
        # if enable_ig_loss:
        #     outer_loop.set_postfix(
        #         train_loss=f'{epoch_fnoloss:.2e}',
        #         ig_loss=f'{epoch_igloss:.2e}',
        #         val_loss=f'{epoch_valloss:.2e}'
        #     )
        # else:
        #     outer_loop.set_postfix(
        #         train_loss=f'{epoch_fnoloss:.2e}',
        #         val_loss=f'{epoch_valloss:.2e}'
        #     )

    cleanup()


# %%
def main(
    enable_ig_loss,
    topo_path, a_path, u_path,
    train_idx, eval_idx,
    case, num_samples_x_y,
    batch_size, epochs, learning_rate,
    scheduler_step, scheduler_gamma,
    operator_type, T_in, T_out, nx, ny,
    magnification_factor,
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

    plot_live_loss = False
    L_x, L_y = 1.0, 1.0

    # Build experiment name (train_size inferred from indices)
    train_size = len(train_idx)
    Mode = (
        f"{case}_{'IG_Enable' if enable_ig_loss else 'IG_Disable'}"
        f"_Nx_{nx}_Ny_{ny}_Tin_{T_in}_Tout_{T_out}"
        f"_Samp_{num_samples_x_y}_{operator_type}"
        f"_DDP_{train_size}"
    )
    print(f"Mode: {Mode}")

    # Paths
    main_path = os.path.join("experiments", case)
    PATH_saved_models = os.path.join(main_path, "saved_models")
    ensure_directory(PATH_saved_models)

    # Model + loss
    # IMPORTANT: passing a *model instance* into spawn can cause pickling issues.
    # Best practice is to construct the model inside each rank (in train_model),
    # but keeping your current pattern for now.

    model_fn = FNO3d(
        T_in=T_in, T_out=T_out,
        modes_x=8, modes_y=8, modes_t=8,
        width=20,
        encoder_kernel_size_x=82,
        encoder_kernel_size_y=41,
        encoder_num_layers=4
    )

    magnifier_fn = magnifier(
        in_channels=2,
        base_channels=32,
        num_fno_blocks=4,
        fno_modes_x=6,
        fno_modes_y=6,
        num_refinement_blocks=4,
        num_residual_per_block=3,
        channel_multipliers=[1.0, 1.5, 2, 2],  # 48→64→80→96→96
        dropout=0.1,
        use_attention=False,
        use_pyramid_pooling=True,
        use_gradient_checkpointing=False
    )

    # Only needed if IG is enabled (still safe to create)
    ss = 2 if enable_ig_loss else 1
    awl_fn = AutomaticWeightedLoss(ss)
    criterion = nn.MSELoss()

    torch.multiprocessing.spawn(
        train_model,
        args=(
            world_size, model_fn, magnifier_fn, awl_fn, learning_rate,
            operator_type, T_in, T_out,
            magnification_factor,
            width_CNO, depth_CNO, kernel_size, unet_depth,
            mode1, mode2, mode3, width_FNO,
            wavelet, level, layers, grid_range, width_WNO,
            branch_layers, trunk_layers,

            epochs, PATH_saved_models, Mode, num_samples_x_y,
            enable_ig_loss, L_x, L_y, criterion, plot_live_loss,
            case, batch_size, topo_path, a_path, u_path, train_idx, eval_idx
        ),
        nprocs=world_size,
        join=True
    )
    # model_fn = create_model(operator_type, T_in, T_out, nx, ny, 
    #                 width_CNO=width_CNO, depth_CNO=depth_CNO, kernel_size=kernel_size, unet_depth=unet_depth,  # CNO inputs
    #                 mode1=mode1, mode2=mode2, mode3=mode3, width_FNO=width_FNO,  # FNO inputs
    #                 wavelet=wavelet, level=level, layers=layers, grid_range=grid_range, width_WNO=width_WNO,  # WNO inputs
    #                 branch_layers=branch_layers, trunk_layers=trunk_layers)

# %%
if __name__ == "__main__":
    # System and environment setup
    # run_nvidia_smi()  # Check GPU status
    # MHPI()           # Initialize MHPI (if this is a custom function)
    import warnings
    warnings.filterwarnings("ignore", message="incompatible copy of pydevd already imported")

    # Dataset and case configuration
    MAIN_PATH = '/storage/group/cxs1024/default/mehdi/Hurricane_Matthew_scenario_groups/scenario_groups_1/'
    # This creates a 'pointer' to the data on disk without filling up your RAM
    topo_path = MAIN_PATH + 'hurricane_matthew_processed_data_bed_2.pt'
    a_path = MAIN_PATH + "hurricane_matthew_processed_data_input_2.pt"
    u_path = MAIN_PATH + "hurricane_matthew_processed_data_solution_2.pt"

    case = 'Hurricane_Matthew'
    enable_ig_loss = False  # Enable/disable IG loss

    # Dataset sizes
    train_size = 300
    eval_size = 150

    train_size = 5
    eval_size = 5


    tmp_ds = LargeHydrologyDataset(a_path, u_path)
    # 2. Split with a fixed generator for reproducibility
    n = len(tmp_ds)

    # Make deterministic indices
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx = perm[:train_size]
    eval_idx  = perm[train_size:train_size + eval_size]

    # Sampling configuration
    num_samples_x_y = 'test_mag'  # Number of random samples along x, y axes for Jacobian calculations

    # Training hyperparameters
    batch_size = 2
    epochs = 500
    learning_rate = 0.005
    scheduler_step = 100
    scheduler_gamma = 0.95

    # Model configuration
    operator_type = 'FNO'  # Options: 'CNO', 'FNO', 'WNO', 'DeepONet'

    # Temporal and spatial dimensions
    total_steps = 89
    T_in = 1
    T_out = total_steps - T_in

    nx = 328
    ny = 164
    magnification_factor = 4
    # Model-specific inputs
    # CNO inputs
    width_CNO, depth_CNO = 128, 4
    kernel_size = 3
    unet_depth = 4

    # FNO inputs
    mode_x = 8
    mode_y = 8
    mode_t = 8
    width_FNO = 20

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
        enable_ig_loss, topo_path, a_path, u_path, train_idx, eval_idx, case, 
        num_samples_x_y, batch_size, 
        epochs, learning_rate, scheduler_step, scheduler_gamma, 
        operator_type, T_in, T_out, nx=nx, ny=ny,
        magnification_factor=magnification_factor, 
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

#%%