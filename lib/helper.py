import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import torch.nn as nn


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
        mode: 'avg' for Average Pooling (mass conservation/blocky)
              'bilinear' for Bilinear Interpolation (smooth/continuous)
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
    elif mode == 'bilinear':
        # Bilinear: Smooth interpolation
        # align_corners=False is usually preferred for downsampling
        x_coarse = F.interpolate(x, size=(nx_new, ny_new), mode='bilinear', align_corners=False)
    else:
        raise ValueError("Mode must be 'avg' or 'bilinear'")
    
    # 4. Reshape and Permute back to [nb, nx_new, ny_new, nt]
    result = x_coarse.view(nb, nt, nx_new, ny_new).permute(0, 2, 3, 1)
    
    return result


    
class LargeHydrologyDataset(Dataset):
    def __init__(self, file_a, file_u, m_map=True):
        # mmap=True keeps data on disk; only indices are loaded initially
        self.m_map = m_map
        self.a = torch.load(file_a, mmap=self.m_map, map_location='cpu')
        self.u = torch.load(file_u, mmap=self.m_map, map_location='cpu')

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        # Data is read from disk into RAM only when indexed
        return self.a[idx], self.u[idx]



