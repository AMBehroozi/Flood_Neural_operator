import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import torch.nn as nn
import gc


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

def plot_ever_inundation_confusion(
    u_true,
    u_pred,
    sample_idx=None,        # if int -> plot that sample; if None -> plot across all samples (mode per pixel)
    inund_th=0.01,
    stride_t=None,          # e.g. 4 to use [..., ::4]; None = no striding
    extent=None,            # e.g. [0, X_range, 0, Y_range]; None = pixel coords
    title_prefix="",
    figsize=(8, 6),
):
    """
    Plot TN/FP/FN/TP confusion map for ever-inundated (max over time > inund_th),
    and PRINT organized confusion counts + 2x2 matrix.

    u_true/u_pred can be torch tensors or numpy arrays.
    Expected shapes:
      - single sample: (nx, ny, nt)
      - many samples:  (N, nx, ny, nt)
    """

    # ---- helpers ----
    def as_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.from_numpy(np.asarray(x))

    def fmt_int(n: int) -> str:
        return f"{n:,}"

    ut = as_tensor(u_true)
    up = as_tensor(u_pred)

    # ---- optional time stride ----
    if stride_t is not None:
        ut = ut[..., ::stride_t]
        up = up[..., ::stride_t]

    # ---- ensure shapes match ----
    if ut.shape != up.shape:
        raise ValueError(f"Shape mismatch: true {tuple(ut.shape)} vs pred {tuple(up.shape)}")

    # ---- colormap ----
    cmap = ListedColormap(["#d9d9d9", "#ff0000", "#66d9ff", "#08306b"])  # TN, FP, FN, TP
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    legend_patches = [
        mpatches.Patch(color="#d9d9d9", label="TN (Correct dry)"),
        mpatches.Patch(color="#ff0000", label="FP (False alarm)"),
        mpatches.Patch(color="#66d9ff", label="FN (Missed flood)"),
        mpatches.Patch(color="#08306b", label="TP (Correct wet)"),
    ]

    # ---- compute & plot ----
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if sample_idx is not None:
        if ut.ndim == 4:
            ut_s = ut[sample_idx]
            up_s = up[sample_idx]
        elif ut.ndim == 3:
            ut_s = ut
            up_s = up
        else:
            raise ValueError(f"Unsupported ndim for sample plot: {ut.ndim}")

        true_max = torch.max(ut_s, dim=-1).values
        pred_max = torch.max(up_s, dim=-1).values

        true_wet = true_max > inund_th
        pred_wet = pred_max > inund_th

        tn = (~true_wet) & (~pred_wet)
        fp = (~true_wet) & ( pred_wet)
        fn = ( true_wet) & (~pred_wet)
        tp = ( true_wet) & ( pred_wet)

        # map codes
        codes = torch.zeros_like(true_wet, dtype=torch.uint8)
        codes[fp] = 1
        codes[fn] = 2
        codes[tp] = 3
        img = codes.detach().cpu().numpy()

        ax.imshow(img.T, extent=extent, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"{title_prefix}Ever-inundation confusion (sample={sample_idx}, th={inund_th} m)", fontsize=14)

        TN = int(tn.sum().item())
        FP = int(fp.sum().item())
        FN = int(fn.sum().item())
        TP = int(tp.sum().item())

        header = f"Confusion counts (sample={sample_idx}, th={inund_th} m)"
        total = TN + FP + FN + TP

    else:
        if ut.ndim != 4:
            raise ValueError("For across-samples plot, expected shape (N, nx, ny, nt).")

        true_max = torch.max(ut, dim=-1).values
        pred_max = torch.max(up, dim=-1).values

        true_wet = true_max > inund_th
        pred_wet = pred_max > inund_th

        tn = (~true_wet) & (~pred_wet)
        fp = (~true_wet) & ( pred_wet)
        fn = ( true_wet) & (~pred_wet)
        tp = ( true_wet) & ( pred_wet)

        TN = int(tn.sum().item())
        FP = int(fp.sum().item())
        FN = int(fn.sum().item())
        TP = int(tp.sum().item())

        header = f"Confusion counts (ALL samples + pixels, th={inund_th} m)"
        total = TN + FP + FN + TP

        # mode map
        counts = torch.stack([tn.sum(0), fp.sum(0), fn.sum(0), tp.sum(0)], dim=0)
        mode_map = torch.argmax(counts, dim=0).to(torch.uint8).detach().cpu().numpy()

        ax.imshow(mode_map.T, extent=extent, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"{title_prefix}Across-samples confusion (MODE per pixel, th={inund_th} m)", fontsize=14)

    # ---- add legend (requested change) ----
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        frameon=True,
        fontsize=10,
        title="Confusion classes",
        title_fontsize=10,
    )

    # --- conditional percentage printout (NOT over all pixels) ---
    cm = np.array([[TN, FP],
                   [FN, TP]], dtype=np.int64)

    def pct(num, den):
        return 100.0 * num / den if den > 0 else float("nan")

    true_wet = TP + FN
    true_dry = TN + FP
    pred_wet = TP + FP
    pred_dry = TN + FN
    total    = TP + TN + FP + FN

    wet_recall     = TP / true_wet if true_wet > 0 else float("nan")   # TPR / POD
    wet_precision  = TP / pred_wet if pred_wet > 0 else float("nan")   # PPV
    dry_recall     = TN / true_dry if true_dry > 0 else float("nan")   # TNR / Specificity
    fpr            = FP / true_dry if true_dry > 0 else float("nan")   # False alarm rate
    fnr            = FN / true_wet if true_wet > 0 else float("nan")   # Miss rate
    csi            = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else float("nan")  # IoU / CSI

    # F1 (wet class)
    f1 = (2 * wet_precision * wet_recall / (wet_precision + wet_recall)
          if np.isfinite(wet_precision) and np.isfinite(wet_recall) and (wet_precision + wet_recall) > 0
          else float("nan"))

    print("\n" + header)
    print("-" * len(header))
    print(f"TP (Correct wet)      : {fmt_int(TP)}")
    print(f"TN (Correct dry)      : {fmt_int(TN)}")
    print(f"FP (False alarm wet)  : {fmt_int(FP)}")
    print(f"FN (Missed wet)       : {fmt_int(FN)}")
    print(f"True wet  (TP+FN)     : {fmt_int(true_wet)}")
    print(f"True dry  (TN+FP)     : {fmt_int(true_dry)}")
    print(f"Pred wet  (TP+FP)     : {fmt_int(pred_wet)}")
    print(f"Pred dry  (TN+FN)     : {fmt_int(pred_dry)}")
    print(f"Total pixels          : {fmt_int(total)}")
    print()
    print(f"Wet Recall / POD  (TP/(TP+FN)) : {wet_recall:.4f}  ({pct(TP, true_wet):6.2f}%)")
    print(f"Wet Precision (TP/(TP+FP))    : {wet_precision:.4f}  ({pct(TP, pred_wet):6.2f}%)")
    print(f"Dry Recall / TNR  (TN/(TN+FP)) : {dry_recall:.4f}  ({pct(TN, true_dry):6.2f}%)")
    print(f"\nOveral performance\n")
    print(f"False Alarm Rate  (FP/(TN+FP)) : {fpr:.4f}  ({pct(FP, true_dry):6.2f}%)")
    print(f"Miss Rate         (FN/(TP+FN)) : {fnr:.4f}  ({pct(FN, true_wet):6.2f}%)")
    print(f"CSI / IoU         (TP/(TP+FP+FN)) : {csi:.4f}")
    print(f"F1-score (wet)     (2PR/(P+R))   : {f1:.4f}")

    return fig, ax



def get_checkpoint_path(case, nx, ny, t_in, t_out, tag, op_type, train_size, ig_enabled, SAVED_MODEL_PATH, if_best=False):
    """Constructs the standard checkpoint filename and path."""
    ig_tag = "IG_Enable" if ig_enabled else "IG_Disable"
    parts = [
        f"saved_model_{case}",
        ig_tag,
        f"Nx_{nx}", f"Ny_{ny}",
        f"Tin_{t_in}", f"Tout_{t_out}",
        f"Samp_{tag}",
        f"{op_type}_DDP_{train_size}"
    ]
    if if_best:
        filename = "_".join(parts) + "_best.pth"
        mode = "_".join(parts[1:]) + '(best)'
    else:
        filename = "_".join(parts) + ".pth"
        mode = "_".join(parts[1:])
    return os.path.join(SAVED_MODEL_PATH, filename), mode



class BathtubReconstructor(nn.Module):
    def __init__(self, topo_patch, f, max_iters=20):
        super().__init__()
        self.f = f
        self.max_iters = max_iters
        
        # Ensure topo has 3 dims: [1, Ny_fine, Nx_fine]
        if topo_patch.dim() == 2:
            topo_patch = topo_patch.unsqueeze(0)
        
        # register_buffer ensures this moves to GPU when you call .to('cuda')
        self.register_buffer("topo", topo_patch)

    def forward(self, u_coarse):
        # Dynamically get the device from the input tensor
        device = u_coarse.device 
        bs, n_y, n_x, n_t = u_coarse.shape
        f = self.f
        nf_y, nf_x = n_y * f, n_x * f
        
        # We use .expand(bs, -1, -1) to make sure the single topo buffer 
        # matches the incoming batch size u_coarse without copying memory.
        topo_batch = self.topo.expand(bs, -1, -1)

        # 1. Reshape topo [BS, ny*f, nx*f] -> [BS, ny, f, nx, f]
        topo_folded = topo_batch.view(bs, n_y, f, n_x, f).permute(0, 1, 3, 2, 4)
        z_fine = topo_folded.reshape(bs, n_y, n_x, 1, f*f)
        
        d_target = u_coarse.unsqueeze(-1)

        # Bounds initialization
        h_low = z_fine.min(dim=-1, keepdim=True)[0]
        h_high = z_fine.max(dim=-1, keepdim=True)[0] + d_target + 1e-3

        for _ in range(self.max_iters):
            h_mid = (h_low + h_high) / 2.0
            d_mid = torch.relu(h_mid - z_fine).mean(dim=-1, keepdim=True)
            
            mask = d_mid < d_target
            h_low = torch.where(mask, h_mid, h_low)
            h_high = torch.where(mask, h_high, h_mid)

        h_final = (h_low + h_high) / 2.0
        u_rec = torch.relu(h_final - z_fine) 

        u_rec = u_rec.view(bs, n_y, n_x, n_t, f, f)
        u_rec = u_rec.permute(0, 1, 4, 2, 5, 3).reshape(bs, nf_y, nf_x, n_t)

        return u_rec


class PaddedIndexProvider:
    def __init__(self, mx, my, N, batch_size=32, subset_fraction=1.0):
        self.padding = N
        self.effective_mx = mx + 2 * self.padding
        self.effective_my = my + 2 * self.padding
        self.N = N
        self.batch_size = batch_size
        self.subset_fraction = subset_fraction # P% e.g. 0.2
        
        self.max_x = self.effective_mx - N
        self.max_y = self.effective_my - N
        self.stride = max(1, N)
        
        self.x_bases = np.arange(0, self.max_x + 1, self.stride)
        self.y_bases = np.arange(0, self.max_y + 1, self.stride)
        
        xv, yv = np.meshgrid(self.x_bases, self.y_bases)
        self.base_coords = torch.stack([
            torch.from_numpy(xv.flatten()).float(),
            torch.from_numpy(yv.flatten()).float()
        ], dim=1)
        
        # Calculate how many windows to pick per epoch
        self.num_total_windows = len(self.base_coords)
        self.num_subset = max(1, int(self.num_total_windows * self.subset_fraction))

    def get_epoch_indices(self):
        shift_x = torch.randint(0, self.stride, (1,)).item()
        shift_y = torch.randint(0, self.stride, (1,)).item()
        
        coords = self.base_coords.clone()
        coords[:, 0] += shift_x
        coords[:, 1] += shift_y
        
        jitter = torch.randint(-1, 2, coords.shape).float()
        coords += jitter
        
        coords[:, 0] = torch.clamp(coords[:, 0], 0, self.max_x)
        coords[:, 1] = torch.clamp(coords[:, 1], 0, self.max_y)
        
        # Shuffle and take only the P% subset
        shuffled = coords[torch.randperm(self.num_total_windows)]
        return shuffled[:self.num_subset].long()

    def get_batches(self):
        indices = self.get_epoch_indices()
        for i in range(0, len(indices), self.batch_size):
            yield indices[i : i + self.batch_size]

# def prepare_patch_input(
#     coarse_u,          # Input: Coarse u from global model [nb, mx, my, nt]
#     fine_bed,          # Input: Full high-resolution bed topography [nb, nx, ny]
#     i_start,           # Starting row index in coarse grid (int)
#     j_start,           # Starting column index in coarse grid (int)
#     N,                 # Coarse patch size (N x N coarse pixels)
#     f,                 # Upscaling factor (each coarse pixel becomes f x f fine pixels)
#     device             # Device to place the final tensor on ('cuda' or 'cpu')
# ):
#     """
#     Prepares one N x N coarse patch for input to the magnifier model.
    
#     What it does:
#     1. Extracts an N x N patch from the coarse u.
#     2. Upsamples (interpolates) that patch to fine resolution: N*f x N*f.
#     3. Extracts the matching fine-resolution bed patch.
#     4. Broadcasts the static bed patch across the time dimension (nt).
#     5. Concatenates interpolated coarse u + fine bed along the channel dimension.
    
#     Returns:
#         patch_input: [nb, 2, P_fine, P_fine, nt]
#         where P_fine = N * f
#         Channel 0: interpolated coarse u (low-frequency guide)
#         Channel 1: fine-resolution bed (static, high-detail local info)
#     """
#     # Get batch size and time dimension from coarse u
#     nb = coarse_u.shape[0]
#     nt = coarse_u.shape[-1]

#     # Calculate fine patch spatial size
#     P_fine = N * f

#     # Step 1: Extract the N x N coarse u patch
#     # Result shape: [nb, N, N, nt]
#     coarse_patch = coarse_u[:, i_start:i_start + N, j_start:j_start + N, :]

#     # Step 2: Interpolate coarse patch to fine resolution
#     # Permute to [nb, nt, N, N] so we can interpolate spatial dims only
#     coarse_patch = coarse_patch.permute(0, 3, 1, 2)  # [nb, nt, N, N]

#     # Bilinear upsampling (scale_factor applies only to spatial dims)
#     # Note: scale_factor cast to float to avoid type issues in some PyTorch versions
#     interp_u = F.interpolate(
#         coarse_patch,
#         scale_factor=(float(f), float(f)),
#         mode='bilinear',
#         align_corners=False
#     )

#     # Back to original order: [nb, P_fine, P_fine, nt]
#     interp_u = interp_u.permute(0, 2, 3, 1)

#     # Step 3: Extract corresponding fine bed patch
#     # Fine starting indices = coarse start * upscaling factor
#     i_f = i_start * f
#     j_f = j_start * f

#     # Slice the fine bed [nb, nx, ny] → [nb, P_fine, P_fine]
#     bed_patch = fine_bed[:, i_f:i_f + P_fine, j_f:j_f + P_fine]

#     # Step 4: Make bed time-aware (static bed, repeat across nt)
#     # Result: [nb, P_fine, P_fine, nt]
#     bed_patch = bed_patch.unsqueeze(-1).expand(-1, -1, -1, nt)

#     # Step 5: Combine inputs along channel dimension
#     # interp_u.unsqueeze(1) → [nb, 1, P_fine, P_fine, nt]
#     # bed_patch.unsqueeze(1) → [nb, 1, P_fine, P_fine, nt]
#     patch_input = torch.cat([interp_u.unsqueeze(1), bed_patch.unsqueeze(1)], dim=1)
#     # Final shape: [nb, 2, P_fine, P_fine, nt]

#     # Move to target device
#     return patch_input.to(device)


def prepare_patch_input(
    coarse_u,          # [nb, mx, my, nt]
    fine_bed,          # [nb, nx, ny]
    i_start,           # Starting row index in coarse grid
    j_start,           # Starting col index in coarse grid
    N,                 # Coarse patch size
    f,                 # Upscaling factor
    device,            # Target device
    u_bathtub=None     # Optional: Global bathtub reconstruction [nb, nx, ny, nt]
):
    """
    Prepares an N*f x N*f patch for the magnifier model.
    If u_bathtub is provided, it returns 3 channels; otherwise, it returns 2.
    """
    nb = coarse_u.shape[0]
    nt = coarse_u.shape[-1]
    P_fine = N * f

    # 1. Extract and Interpolate Coarse Patch
    # [nb, N, N, nt] -> [nb, nt, N, N] -> [nb, nt, Pf, Pf] -> [nb, Pf, Pf, nt]
    coarse_patch = coarse_u[:, i_start:i_start + N, j_start:j_start + N, :]
    interp_u = F.interpolate(
        coarse_patch.permute(0, 3, 1, 2),
        scale_factor=float(f),
        mode='bilinear',
        align_corners=False
    ).permute(0, 2, 3, 1)

    # 2. Extract Fine Bed Patch
    i_f, j_f = i_start * f, j_start * f
    bed_patch = fine_bed[:, i_f:i_f + P_fine, j_f:j_f + P_fine]
    # Static bed expanded across time: [nb, Pf, Pf, nt]
    bed_patch = bed_patch.unsqueeze(-1).expand(-1, -1, -1, nt)

    # 3. Assemble Channel List
    # Every channel is unsqueezed to [nb, 1, Pf, Pf, nt]
    channels = [interp_u.unsqueeze(1), bed_patch.unsqueeze(1)]

    # 4. Handle Optional Bathtub Input (The Augmentation)
    if u_bathtub is not None:
        # Slice the matching HR patch from the global bathtub field
        # Shape: [nb, Pf, Pf, nt]
        bt_patch = u_bathtub[:, i_f:i_f + P_fine, j_f:j_f + P_fine, :]
        channels.append(bt_patch.unsqueeze(1))

    # 5. Final Concatenation
    # If u_bathtub is None: [nb, 2, Pf, Pf, nt]
    # If u_bathtub is present: [nb, 3, Pf, Pf, nt]
    patch_input = torch.cat(channels, dim=1)

    return patch_input.to(device)



def coarsen_spatial_tensor(tensor, N, mode='avg'):
    """
    Coarsens a tensor of shape [nb, nx, ny, nt] by a factor of N.
    
    Args:
        tensor: Input tensor [nb, nx, ny, nt]
        N: Downsampling factor
        mode: 'avg' for Average Pooling
              'bilinear' for Bilinear Interpolation
              'area' for Area Interpolation (Best for Mass Conservation)
    """
    if N == 1:
        return tensor
        
    nb, nx, ny, nt = tensor.shape
    
    # 1. Prepare for 2D spatial operations
    # Shape: [nb * nt, 1, nx, ny]
    x = tensor.permute(0, 3, 1, 2).reshape(nb * nt, 1, nx, ny)
    
    # 2. Calculate new dimensions
    nx_new, ny_new = nx // N, ny // N

    # 3. Apply selected method
    if mode == 'avg':
        # Average Pooling: Discrete blocks
        x_coarse = F.avg_pool2d(x, kernel_size=N, stride=N)
    elif mode == 'area':
        # Area: Resamples using pixel area relation (excellent for downsampling)
        x_coarse = F.interpolate(x, size=(nx_new, ny_new), mode='area')
    elif mode == 'bilinear':
        # Bilinear: Smooth interpolation
        x_coarse = F.interpolate(x, size=(nx_new, ny_new), mode='bilinear', align_corners=False)
    else:
        raise ValueError("Mode must be 'avg', 'area', or 'bilinear'")
    
    # 4. Reshape and Permute back to [nb, nx_new, ny_new, nt]
    result = x_coarse.view(nb, nt, nx_new, ny_new).permute(0, 2, 3, 1)
    
    return result

# def coarsen_spatial_tensor(tensor, N, mode='avg'):
#     """
#     Coarsens a tensor of shape [nb, nx, ny, nt] by a factor of N.
    
#     Args:
#         tensor: Input tensor [nb, nx, ny, nt]
#         N: Downsampling factor
#         mode: 'avg' for Average Pooling (mass conservation/blocky)
#               'bilinear' for Bilinear Interpolation (smooth/continuous)
#     """
#     if N == 1:
#         return tensor
        
#     nb, nx, ny, nt = tensor.shape
    
#     # 1. Prepare for 2D spatial operations
#     # Shape: [nb * nt, 1, nx, ny]
#     x = tensor.permute(0, 3, 1, 2).reshape(nb * nt, 1, nx, ny)
    
#     # 2. Calculate new dimensions
#     nx_new, ny_new = nx // N, ny // N

#     # 3. Apply selected method
#     if mode == 'avg':
#         # Average Pooling: Discrete blocks
#         x_coarse = F.avg_pool2d(x, kernel_size=N, stride=N)
#     elif mode == 'bilinear':
#         # Bilinear: Smooth interpolation
#         # align_corners=False is usually preferred for downsampling
#         x_coarse = F.interpolate(x, size=(nx_new, ny_new), mode='bilinear', align_corners=False)
#     else:
#         raise ValueError("Mode must be 'avg' or 'bilinear'")
    
#     # 4. Reshape and Permute back to [nb, nx_new, ny_new, nt]
#     result = x_coarse.view(nb, nt, nx_new, ny_new).permute(0, 2, 3, 1)
    
#     return result

# class LargeHydrologyDataset(Dataset):
#     def __init__(self, file_a, file_u, m_map=True):
#         # mmap=True keeps data on disk; only indices are loaded initially
#         self.m_map = m_map
#         self.a = torch.load(file_a, mmap=self.m_map, map_location='cpu')
#         self.u = torch.load(file_u, mmap=self.m_map, map_location='cpu')

#     def __len__(self):
#         return self.a.shape[0]

#     def __getitem__(self, idx):
#         # Data is read from disk into RAM only when indexed
#         return self.a[idx], self.u[idx]



import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from matplotlib.path import Path

class LargeHydrologyDataset(Dataset):
    def __init__(self, file_a, file_u, m_map=True, mask=False, csv_path=None, Lx=None, Ly=None):
        self.m_map = m_map
        self.a = torch.load(file_a, mmap=self.m_map, map_location='cpu')
        self.u = torch.load(file_u, mmap=self.m_map, map_location='cpu')
        
        self.do_masking = mask
        self.spatial_mask = None

        if self.do_masking:
            if csv_path is None or Lx is None or Ly is None:
                raise ValueError("Masking requires csv_path, Lx, and Ly to be defined.")
            
            # Pre-compute the mask during initialization
            self.spatial_mask = self._generate_static_mask(csv_path, Lx, Ly)

    def _generate_static_mask(self, csv_path, Lx, Ly):
        """Creates a [nx, ny, 1] mask based on boundary points."""
        # Load nx, ny from the loaded data
        _, nx, ny, _ = self.a.shape
        
        # Load raw CSV (no headers)
        df = pd.read_csv(csv_path, header=None)
        points = df[[0, 1]].values 
        
        # Create coordinate grid
        x_coords = np.linspace(0, Lx, nx)
        y_coords = np.linspace(0, Ly, ny)
        xv, yv = np.meshgrid(x_coords, y_coords, indexing='ij')
        grid_points = np.vstack((xv.flatten(), yv.flatten())).T
        
        # Point-in-polygon logic
        poly_path = Path(points)
        mask_flat = poly_path.contains_points(grid_points)
        
        # Convert to tensor and reshape for broadcasting [nx, ny, 1]
        mask_2d = torch.from_numpy(mask_flat.reshape(nx, ny))
        final_mask = torch.logical_not(mask_2d).float()
        return final_mask.unsqueeze(-1) # Shape: [nx, ny, 1]

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        # Fetch data (mmap keeps this efficient)
        sample_a = self.a[idx]
        sample_u = self.u[idx]

        # Apply mask if requested
        if self.do_masking and self.spatial_mask is not None:
            sample_u = sample_u * self.spatial_mask

        return sample_a, sample_u