import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

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
from lib.helper import LargeHydrologyDataset
from lib.ddp_helpers import setup, cleanup
from lib.helper import coarsen_spatial_tensor
import torch.nn.functional as F

# If you enable IG later, you will need these:
# from lib.low_rank_jacobian import compute_low_rank_jacobian_1, compute_low_rank_jacobian_loss


#
class PaddedIndexProvider:
    def __init__(self, mx, my, N, batch_size=32):
        # We increase the 'effective' domain so windows can overlap the real edges
        self.padding = N
        self.effective_mx = mx + 2 * self.padding
        self.effective_my = my + 2 * self.padding
        self.N = N
        self.batch_size = batch_size
        
        # Valid starts in the padded coordinate system
        self.max_x = self.effective_mx - N
        self.max_y = self.effective_my - N
        
        self.stride = max(1, N // 2)
        
        # Grid covers the entire padded area
        self.x_bases = np.arange(0, self.max_x + 1, self.stride)
        self.y_bases = np.arange(0, self.max_y + 1, self.stride)
        
        xv, yv = np.meshgrid(self.x_bases, self.y_bases)
        self.base_coords = torch.stack([
            torch.from_numpy(xv.flatten()).float(),
            torch.from_numpy(yv.flatten()).float()
        ], dim=1)
        
        self.num_total_windows = len(self.base_coords)

    def get_epoch_indices(self):
        shift_x = torch.randint(0, self.stride, (1,)).item()
        shift_y = torch.randint(0, self.stride, (1,)).item()
        
        coords = self.base_coords.clone()
        coords[:, 0] += shift_x
        coords[:, 1] += shift_y
        
        # Local jitter
        jitter = torch.randint(-1, 2, coords.shape).float()
        coords += jitter
        
        # Hard clamp to ensure indices are ALWAYS positive and safe for slicing
        coords[:, 0] = torch.clamp(coords[:, 0], 0, self.max_x)
        coords[:, 1] = torch.clamp(coords[:, 1], 0, self.max_y)
        
        return coords[torch.randperm(self.num_total_windows)].long()


    def get_batches(self):
        indices = self.get_epoch_indices()
        for i in range(0, self.num_total_windows, self.batch_size):
            yield indices[i : i + self.batch_size]


def prepare_patch_input(
    coarse_u,          # Input: Coarse u from global model [nb, mx, my, nt]
    fine_bed,          # Input: Full high-resolution bed topography [nb, nx, ny]
    i_start,           # Starting row index in coarse grid (int)
    j_start,           # Starting column index in coarse grid (int)
    N,                 # Coarse patch size (N x N coarse pixels)
    f,                 # Upscaling factor (each coarse pixel becomes f x f fine pixels)
    device             # Device to place the final tensor on ('cuda' or 'cpu')
):
    """
    Prepares one N x N coarse patch for input to the magnifier model.
    
    What it does:
    1. Extracts an N x N patch from the coarse u.
    2. Upsamples (interpolates) that patch to fine resolution: N*f x N*f.
    3. Extracts the matching fine-resolution bed patch.
    4. Broadcasts the static bed patch across the time dimension (nt).
    5. Concatenates interpolated coarse u + fine bed along the channel dimension.
    
    Returns:
        patch_input: [nb, 2, P_fine, P_fine, nt]
        where P_fine = N * f
        Channel 0: interpolated coarse u (low-frequency guide)
        Channel 1: fine-resolution bed (static, high-detail local info)
    """
    # Get batch size and time dimension from coarse u
    nb = coarse_u.shape[0]
    nt = coarse_u.shape[-1]

    # Calculate fine patch spatial size
    P_fine = N * f

    # Step 1: Extract the N x N coarse u patch
    # Result shape: [nb, N, N, nt]
    coarse_patch = coarse_u[:, i_start:i_start + N, j_start:j_start + N, :]

    # Step 2: Interpolate coarse patch to fine resolution
    # Permute to [nb, nt, N, N] so we can interpolate spatial dims only
    coarse_patch = coarse_patch.permute(0, 3, 1, 2)  # [nb, nt, N, N]

    # Bilinear upsampling (scale_factor applies only to spatial dims)
    # Note: scale_factor cast to float to avoid type issues in some PyTorch versions
    interp_u = F.interpolate(
        coarse_patch,
        scale_factor=(float(f), float(f)),
        mode='bilinear',
        align_corners=False
    )

    # Back to original order: [nb, P_fine, P_fine, nt]
    interp_u = interp_u.permute(0, 2, 3, 1)

    # Step 3: Extract corresponding fine bed patch
    # Fine starting indices = coarse start * upscaling factor
    i_f = i_start * f
    j_f = j_start * f

    # Slice the fine bed [nb, nx, ny] → [nb, P_fine, P_fine]
    bed_patch = fine_bed[:, i_f:i_f + P_fine, j_f:j_f + P_fine]

    # Step 4: Make bed time-aware (static bed, repeat across nt)
    # Result: [nb, P_fine, P_fine, nt]
    bed_patch = bed_patch.unsqueeze(-1).expand(-1, -1, -1, nt)

    # Step 5: Combine inputs along channel dimension
    # interp_u.unsqueeze(1) → [nb, 1, P_fine, P_fine, nt]
    # bed_patch.unsqueeze(1) → [nb, 1, P_fine, P_fine, nt]
    patch_input = torch.cat([interp_u.unsqueeze(1), bed_patch.unsqueeze(1)], dim=1)
    # Final shape: [nb, 2, P_fine, P_fine, nt]

    # Move to target device
    return patch_input.to(device)


def DummyMagnifier(patch_input):
    """
    Input:
        patch_input: [nb_total, 2, P_fine, P_fine, nt]
    Output:
        refined_u:   [nb_total, 1, P_fine, P_fine, nt]
    """
    conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
    # Apply a dummy operation to mimic refinement
    # We use Conv3d to maintain the temporal and spatial relationships
    refined_u = conv(patch_input)
    return refined_u


#


def train_model(rank, world_size, model_fn, awl_fn, learning_rate, 
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
    model = DDP(model, device_ids=[rank])

    # ---- Optimizer (AWL only if IG enabled) ----
    if enable_ig_loss:
        awl = awl_fn.to(rank)
        awl = DDP(awl, device_ids=[rank])
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'lr': learning_rate},
             {'params': awl.parameters(),   'lr': learning_rate}]
        )
    else:
        awl = None
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
    
    nx = None
    ny = None

    N = 5
    f = magnification_factor

    index_provider = PaddedIndexProvider(mx=82, my=41, N=5, batch_size=32)
    # Assuming model2, optimizer2, and index_provider are initialized per rank
    for ep in outer_loop:
        train_sampler.set_epoch(ep)
        
        model.train()   # Model 1
        # model2.train()  # Magnifier
        
        total_fnoloss = 0.0
        total_igloss  = 0.0
        total_magloss = 0.0  # Loss for Magnifier
        total_samples = 0

        # Get fresh randomized spatial sweep for this rank/epoch
        # epoch_indices: [Total_Windows, 2] (Starts in Padded Coordinate System)
        epoch_indices = index_provider.get_epoch_indices()

        for batch_data in train_loader:
            batch_data = [item.to(rank) for item in batch_data]

            # batch_forcing: [bs, nx, ny, nt]
            # batch_u0:      [bs, nx, ny, T_in]
            # batch_u_out_hr:[bs, nx, ny, T_out] (Original High-Res Ground Truth)
            batch_forcing  = batch_data[0]
            batch_u0       = batch_data[1][..., :T_in]
            batch_u_out_hr = batch_data[1][..., T_in:] 
            
            bs = batch_u0.shape[0]
            total_samples += bs
            batch_topo = topo.expand(bs, -1, -1) # [bs, nx, ny]

            if nx is None or ny is None:
                nx, ny = batch_forcing.shape[1:3]

            # 1. ─── GLOBAL COARSE PASS (MODEL 1) ───
            optimizer.zero_grad(set_to_none=True)
            
            # U_pred: [bs, mx, my, T_out]
            U_pred = model(batch_forcing, batch_u0, batch_topo)
            
            # Ground Truth Coarsening for Model 1 Loss
            # batch_u_out_lr: [bs, mx, my, T_out]
            batch_u_out_lr = coarsen_spatial_tensor(batch_u_out_hr, N=f, mode='bilinear')
            
            data_loss = criterion(U_pred, batch_u_out_lr)
            
            # ... (IG Loss logic for Model 1 remains the same) ...
            if enable_ig_loss:
                U_mat_pred, V_mat_pred = compute_low_rank_jacobian_1(model, U_pred, ...)
                ig_loss = compute_low_rank_jacobian_loss(...)
                loss1 = awl(data_loss, ig_loss)
            else:
                ig_loss = data_loss.new_tensor(0.0)
                loss1 = data_loss

            # loss1.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # optimizer.step()

            # 2. ─── PATCH REFINEMENT PASS (MODEL 2) ───
            # We use a 'detach' on U_pred to train Model 2 independently if desired, 
            # or keep gradients flowing for end-to-end. Here we use detach for stability.
            
            # PRE-PADDING: Pad global tensors to satisfy PaddedIndexProvider ranges
            # u_pad: [bs, mx + 2N, my + 2N, T_out]
            u_pad = F.pad(U_pred.detach().permute(0, 3, 1, 2), (N, N, N, N), mode='replicate').permute(0, 2, 3, 1)
            # topo_pad: [bs, nx + 2Pf, ny + 2Pf]
            topo_pad = F.pad(batch_topo, (N*f, N*f, N*f, N*f), mode='replicate')
            # target_pad: [bs, nx + 2Pf, ny + 2Pf, T_out]
            target_pad = F.pad(batch_u_out_hr.permute(0, 3, 1, 2), (N*f, N*f, N*f, N*f), mode='replicate').permute(0, 2, 3, 1)

            # Spatial Mini-batching: Process nb_i windows at a time
            for spatial_batch in index_provider.get_batches():
                # spatial_batch: [nb_i, 2]
                
                # optimizer2.zero_grad(set_to_none=True)
                
                batch_patches_in = []
                batch_patches_trgt = []

                for (i_s, j_s) in spatial_batch:
                    # prepare_patch_input returns [bs, 2, Pf, Pf, T_out]
                    p_in = prepare_patch_input(u_pad, topo_pad, i_s, j_s, N, f, rank)
                    
                    # Slicing target from padded high-res ground truth
                    # p_target: [bs, 1, Pf, Pf, T_out]
                    # Change this:
                    p_target = target_pad[:, i_s*f : i_s*f + N*f, j_s*f : j_s*f + N*f, :].unsqueeze(1)              
                    batch_patches_in.append(p_in)
                    batch_patches_trgt.append(p_target)

                # Flattening: Combine bs and nb_i into one large batch dimension
                # big_in: [bs * nb_i, 2, Pf, Pf, T_out]
                big_in = torch.cat(batch_patches_in, dim=0)
                # big_trgt: [bs * nb_i, 1, Pf, Pf, T_out]
                big_trgt = torch.cat(batch_patches_trgt, dim=0)

                # Model 2 Forward
                # big_out: [bs * nb_i, 1, Pf, Pf, T_out]
                big_out = DummyMagnifier(big_in)
                
                mag_loss = criterion(big_out, big_trgt)
                # mag_loss.backward()
                # optimizer2.step()
                
                total_magloss += mag_loss.item() * (bs * len(spatial_batch))

            # Accumulate stats
            total_fnoloss += data_loss.item() * bs
            total_igloss  += ig_loss.item() * bs

        # Epoch Metrics
        epoch_fnoloss = total_fnoloss / total_samples
        epoch_igloss  = total_igloss  / total_samples
        epoch_magloss = total_magloss / (total_samples * index_provider.num_total_windows)

        train_fnolosses.append(epoch_fnoloss)
        train_maglosses.append(epoch_magloss) 
        # Track magnifier progress

        # ---------------- Evaluation phase ----------------
        model.eval()
        total_valloss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for batch_data in eval_loader:
                batch_data = [item.to(rank) for item in batch_data]

                batch_forcing = batch_data[0]
                batch_u0      = batch_data[1][..., :T_in]
                batch_u_out   = batch_data[1][..., T_in:]

                bs = batch_u0.shape[0]
                total_val_samples += bs

                batch_topo = topo.expand(bs, -1, -1)

                U_pred = model(batch_forcing, batch_u0, batch_topo)
                val_loss = criterion(U_pred, batch_u_out)

                total_valloss += val_loss.item() * bs

        epoch_valloss = total_valloss / total_val_samples
        val_losses.append(epoch_valloss)

        # ---------------- Logging / saving ----------------
        losses_dict = {
            'Training FNO Loss': train_fnolosses,
            'Validation Loss': val_losses
        }
        if enable_ig_loss:
            losses_dict['Train IG loss'] = train_iglosses

        df = pd.DataFrame(losses_dict)

        save_results = False
        if save_results and (ep % 5 == 0) and (rank == 0):
            torch.save({
                'config': {
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
                },
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_df': df,          # clearer name
            }, PATH_saved_models + f'/saved_model_{Mode}.pth')

        if plot_live_loss and (ep % 1 == 0):
            loss_live_plot(losses_dict)

        scheduler.step()

        outer_loop.set_description(f"Progress (Epoch {ep + 1}/{epochs}) Mode: {Mode}")
        if enable_ig_loss:
            outer_loop.set_postfix(
                train_loss=f'{epoch_fnoloss:.2e}',
                ig_loss=f'{epoch_igloss:.2e}',
                val_loss=f'{epoch_valloss:.2e}'
            )
        else:
            outer_loop.set_postfix(
                train_loss=f'{epoch_fnoloss:.2e}',
                val_loss=f'{epoch_valloss:.2e}'
            )

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

    # Only needed if IG is enabled (still safe to create)
    ss = 2 if enable_ig_loss else 1
    awl_fn = AutomaticWeightedLoss(ss)
    criterion = nn.MSELoss()

    torch.multiprocessing.spawn(
        train_model,
        args=(
            world_size, model_fn, awl_fn, learning_rate,
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