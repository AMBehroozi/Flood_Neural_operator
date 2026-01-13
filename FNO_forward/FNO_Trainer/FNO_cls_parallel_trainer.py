import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import pandas as pd
from tqdm import tqdm

# Local paths (only if you truly need them)
sys.path.append("../../")
sys.path.append("../")
sys.path.append("./")

from lib.utilities3 import ensure_directory
from lib.utiltools import loss_live_plot, AutomaticWeightedLoss

from models.fno3d_cls import FNO3d
from lib.helper import LargeHydrologyDataset
from lib.ddp_helpers import setup, cleanup
# If you enable IG later, you will need these:
# from lib.low_rank_jacobian import compute_low_rank_jacobian_1, compute_low_rank_jacobian_loss

def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / (mask.sum() + eps)

def train_model(rank, world_size, model_fn, awl_fn, awl2_fn, learning_rate, 
                operator_type, T_in, T_out,

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
        awl2 = awl2_fn.to(rank)
        awl2 = DDP(awl2, device_ids=[rank])

        optimizer = optim.Adam(
            [{'params': model.parameters(), 'lr': learning_rate},
             {'params': awl.parameters(),   'lr': learning_rate},
             {'params': awl2.parameters(),   'lr': learning_rate}]
        )
    else:
        awl2 = awl2_fn.to(rank)
        awl2 = DDP(awl2, device_ids=[rank])
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'lr': learning_rate},
             {'params': awl2.parameters(),   'lr': learning_rate}]
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
    nx = None
    ny = None
    for ep in outer_loop:
        train_sampler.set_epoch(ep)

        # ---------------- Training phase ----------------
        model.train()
        total_fnoloss = 0.0          # will log your regression mix (awl2 output)
        total_igloss  = 0.0
        total_samples = 0

        # thresholds (match your data prep)
        tau_wet = 0.025              # wet/dry threshold in meters
        tau_shallow = 0.5            # shallow-wet band upper bound (tune)
        eps = 1e-6

        for batch_data in train_loader:
            batch_data = [item.to(rank) for item in batch_data]

            batch_forcing = batch_data[0]
            batch_u0      = batch_data[1][..., :T_in]
            batch_u_out   = batch_data[1][..., T_in:]

            bs = batch_u0.shape[0]
            total_samples += bs

            batch_topo = topo.expand(bs, -1, -1)

            if nx is None or ny is None:
                nx, ny = batch_forcing.shape[1:3]

            optimizer.zero_grad(set_to_none=True)

            # model MUST return (U_pred, wet_logits)
            U_pred, wet_logits = model(batch_forcing, batch_u0, batch_topo)

            # ---- masks from TRUE depth ----
            wet_mask     = (batch_u_out > tau_wet)
            shallow_mask = wet_mask & (batch_u_out < tau_shallow)

            err = U_pred - batch_u_out

            # (1) asinh loss on wet pixels
            L_asinh = masked_mean(
                (torch.asinh(U_pred) - torch.asinh(batch_u_out)).abs(),
                wet_mask
            )

            # (2) shallow-focused L1 on shallow wet pixels
            L_shallow = masked_mean(err.abs(), shallow_mask)

            # (3) relative abs error on wet pixels (clamped)
            rel = err.abs() / (batch_u_out.abs() + eps)
            rel = torch.clamp(rel, max=20.0)
            L_rel = masked_mean(rel, wet_mask)

            # (4) BCE wet/dry (pixelwise)
            wet_target = wet_mask.to(dtype=wet_logits.dtype)
            L_bce = F.binary_cross_entropy_with_logits(wet_logits, wet_target)

            # ---- combine 4 regression+classification terms with AWL2 ----
            # awl2 must be AutomaticWeightedLoss(4)
            reg_loss = awl2(L_asinh, L_shallow, L_rel, L_bce)

            # ---- optionally combine with IG using your existing AWL (2-term) ----
            if enable_ig_loss:
                U_mat_pred, V_mat_pred = compute_low_rank_jacobian_1(
                    model, U_pred, batch_parameter, batch_u_in, rank=5, epsilon=1e-1, seed=None
                )
                ig_loss = compute_low_rank_jacobian_loss(
                    du_dp__low_rank_true, U_mat_pred, V_mat_pred, method='action'
                )
                loss = awl(reg_loss, ig_loss)   # awl must be AutomaticWeightedLoss(2)
            else:
                ig_loss = reg_loss.new_tensor(0.0)
                loss = reg_loss

            loss.backward()
            optimizer.step()

            total_fnoloss += reg_loss.item() * bs
            total_igloss  += ig_loss.item() * bs

        epoch_fnoloss = total_fnoloss / total_samples
        epoch_igloss  = total_igloss  / total_samples

        train_fnolosses.append(epoch_fnoloss)
        train_iglosses.append(epoch_igloss)

        
        # ---------------- Evaluation phase ----------------
        model.eval()
        total_valloss = 0.0
        total_val_samples = 0

        # same thresholds as train
        tau_wet = 0.025
        tau_shallow = 0.5
        eps = 1e-6

        with torch.no_grad():
            for batch_data in eval_loader:
                batch_data = [item.to(rank) for item in batch_data]

                batch_forcing = batch_data[0]
                batch_u0      = batch_data[1][..., :T_in]
                batch_u_out   = batch_data[1][..., T_in:]

                bs = batch_u0.shape[0]
                total_val_samples += bs

                batch_topo = topo.expand(bs, -1, -1)

                # model returns both now
                U_pred, wet_logits = model(batch_forcing, batch_u0, batch_topo)

                # ---- same composite loss as training (NO IG in eval) ----
                wet_mask     = (batch_u_out > tau_wet)
                shallow_mask = wet_mask & (batch_u_out < tau_shallow)

                err = U_pred - batch_u_out

                L_asinh = masked_mean(
                    (torch.asinh(U_pred) - torch.asinh(batch_u_out)).abs(),
                    wet_mask
                )
                L_shallow = masked_mean(err.abs(), shallow_mask)

                rel = err.abs() / (batch_u_out.abs() + eps)
                rel = torch.clamp(rel, max=20.0)
                L_rel = masked_mean(rel, wet_mask)

                wet_target = wet_mask.to(dtype=wet_logits.dtype)
                L_bce = F.binary_cross_entropy_with_logits(wet_logits, wet_target)

                # IMPORTANT: use awl2 (4-term) here too, so train/val are comparable
                val_loss = awl2(L_asinh, L_shallow, L_rel, L_bce)

                total_valloss += val_loss.item() * bs

        epoch_valloss = total_valloss / max(1, total_val_samples)
        val_losses.append(epoch_valloss)


        # ---------------- Logging / saving ----------------
        losses_dict = {
            "Training FNO Loss": train_fnolosses,   # (this is your reg_loss/awl2 in train)
            "Validation Loss": val_losses,
        }
        if enable_ig_loss:
            losses_dict["Train IG loss"] = train_iglosses

        df = pd.DataFrame(losses_dict)

        save_results = True
        if save_results and (ep % 5 == 0) and (rank == 0):
            torch.save(
                {
                    "config": {
                        "operator_type": operator_type,
                        "enable_ig_loss": enable_ig_loss,
                        "Nx": nx,
                        "Ny": ny,
                        "T_in": T_in,
                        "T_out": T_out,
                        "tau_wet": tau_wet,
                        "tau_shallow": tau_shallow,

                        "width_CNO": width_CNO,
                        "depth_CNO": depth_CNO,
                        "kernel_size": kernel_size,
                        "unet_depth": unet_depth,

                        "mode1": mode1,
                        "mode2": mode2,
                        "mode3": mode3,
                        "width_FNO": width_FNO,

                        "wavelet": wavelet,
                        "level": level,
                        "layers": layers,
                        "grid_range": grid_range,
                        "width_WNO": width_WNO,

                        "branch_layers": branch_layers,
                        "trunk_layers": trunk_layers,
                    },
                    "epoch": ep,

                    # if this was trained under DDP, saving model.module is cleaner
                    "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),

                    # optional but helpful if you ever resume
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss_df": df,
                },
                os.path.join(PATH_saved_models, f"saved_model_{Mode}.pth"),
            )

        if plot_live_loss and (ep % 1 == 0):
            loss_live_plot(losses_dict)

        scheduler.step()

        outer_loop.set_description(f"Progress (Epoch {ep + 1}/{epochs}) Mode: {Mode}")
        if enable_ig_loss:
            outer_loop.set_postfix(
                train_loss=f"{epoch_fnoloss:.2e}",
                ig_loss=f"{epoch_igloss:.2e}",
                val_loss=f"{epoch_valloss:.2e}",
            )
        else:
            outer_loop.set_postfix(
                train_loss=f"{epoch_fnoloss:.2e}",
                val_loss=f"{epoch_valloss:.2e}",
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
        encoder_kernel_size_x=100,
        encoder_kernel_size_y=50,
        encoder_num_layers=4
    )

    # Only needed if IG is enabled (still safe to create)
    ss = 2 if enable_ig_loss else 1
    awl_fn = AutomaticWeightedLoss(ss)
    awl2_fn = AutomaticWeightedLoss(4)
    criterion = nn.MSELoss()

    torch.multiprocessing.spawn(
        train_model,
        args=(
            world_size, model_fn, awl_fn, awl2_fn, learning_rate,
            operator_type, T_in, T_out,

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
    topo_path = MAIN_PATH + 'hurricane_matthew_processed_data_bed.pt'
    a_path = MAIN_PATH + "hurricane_matthew_processed_data_input.pt"
    u_path = MAIN_PATH + "hurricane_matthew_processed_data_solution.pt"

    case = 'Hurricane_Matthew'
    enable_ig_loss = False  # Enable/disable IG loss

    # Dataset sizes
    train_size = 300
    eval_size = 150



    tmp_ds = LargeHydrologyDataset(a_path, u_path)
    # 2. Split with a fixed generator for reproducibility
    n = len(tmp_ds)

    # Make deterministic indices
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx = perm[:train_size]
    eval_idx  = perm[train_size:train_size + eval_size]

    # Sampling configuration
    num_samples_x_y = 'test_multi_loss'  # Number of random samples along x, y axes for Jacobian calculations

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

    nx = 313
    ny = 158

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