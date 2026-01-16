import os
import sys
import glob
from netCDF4 import Dataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F


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



def get_subfolders(path):
    # Check if the path actually exists first to avoid errors
    if os.path.exists(path):
        # os.listdir(path) gets everything (files and folders)
        # os.path.isdir checks if the item is specifically a directory
        subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        return subfolders
    else:
        print(f"Error: Path {path} does not exist.")
        return []

def get_sww_mesh_and_states(sww_path, last_index=-1, dry_threshold=0.025):
    """
    Reads SWW file and enforces a minimum depth threshold.
    Cells with depth < dry_threshold are forced to stage = elevation and zero momentum.
    """
    if not os.path.exists(sww_path):
        raise FileNotFoundError(f"SWW file not found at: {sww_path}")

    nc = Dataset(sww_path, 'r')

    # 1. EXTRACT VERTICES
    v_x = nc.variables['x'][:]
    v_y = nc.variables['y'][:]
    vertices = np.column_stack((v_x, v_y))
    
    # 2. EXTRACT CONNECTIVITY
    triangles = nc.variables['volumes'][:]
    
    # 3. EXTRACT STATES
    stage = nc.variables['stage'][last_index, :].copy() # Use .copy() to allow modification
    xmom  = nc.variables['xmomentum'][last_index, :].copy()
    ymom  = nc.variables['ymomentum'][last_index, :].copy()
    
    if nc.variables['elevation'].ndim == 2:
        elev = nc.variables['elevation'][last_index, :]
    else:
        elev = nc.variables['elevation'][:]
    
    # 4. ENFORCE DRY THRESHOLD
    # Calculate depth: h = w - z
    depth = stage - elev
    
    # Create a mask for dry cells
    dry_indices = depth < dry_threshold
    
    # Force stage to match elevation (depth becomes 0.0)
    stage[dry_indices] = elev[dry_indices]
    
    # Force momentum to zero in dry cells to prevent "phantom" flows
    xmom[dry_indices] = 0.0
    ymom[dry_indices] = 0.0

    time_last = nc.variables['time'][last_index]
    nc.close()

    print(f"✓ Loaded states. Forced {np.sum(dry_indices)} cells to dry state (threshold: {dry_threshold}m)")

    return {
        'vertices': vertices,
        'triangles': triangles,
        'stage': stage,
        'elevation': elev,
        'xmomentum': xmom,
        'ymomentum': ymom,
        'time_last': time_last
    }

# def get_sww_mesh_and_states(sww_path, last_index=-1):
#     if not os.path.exists(sww_path):
#         raise FileNotFoundError(f"SWW file not found at: {sww_path}")

#     nc = Dataset(sww_path, 'r')

#     # 1. EXTRACT VERTICES (The corners of the triangles)
#     # In standard ANUGA SWW, 'x' and 'y' are the vertex coordinates
#     v_x = nc.variables['x'][:]
#     v_y = nc.variables['y'][:]
#     vertices = np.column_stack((v_x, v_y))
    
#     # 2. EXTRACT CONNECTIVITY
#     # 'volumes' maps vertex indices to triangles
#     triangles = nc.variables['volumes'][:]
    
#     # 3. EXTRACT STATES (Centroid values)
#     # Note: Ensure you are using the correct variable names. 
#     # Usually 'stage', 'xmomentum', etc. are stored at centroids.
#     stage = nc.variables['stage'][last_index, :]
#     xmom  = nc.variables['xmomentum'][last_index, :]
#     ymom  = nc.variables['ymomentum'][last_index, :]
#     elev  = nc.variables['elevation'][last_index, :] if nc.variables['elevation'].ndim == 2 else nc.variables['elevation'][:]
#     time_last = nc.variables['time'][last_index]
#     nc.close()

#     return {
#         'vertices': vertices,
#         'triangles': triangles,
#         'stage': stage,
#         'elevation': elev,
#         'xmomentum': xmom,
#         'ymomentum': ymom,
#         'time_last': time_last
#     }


# Function to merge parallel ANUGA SWW files
def merge_sww_files(directory='results', output_name='merged.sww', 
                           delete_originals=False, verbose=True):
    """
    Merge parallel ANUGA files, weld duplicate nodes, and optionally delete originals.
    """
    pattern = os.path.join(directory, '*_P*.sww')
    sww_files = sorted(glob.glob(pattern))
    
    if not sww_files:
        raise FileNotFoundError(f"No partial SWW files found in {directory}")

    if verbose:
        print("=" * 60)
        print(f"MERGING & WELDING {len(sww_files)} FILES")
        print("=" * 60)

    # 1. Collect all raw data from processors
    all_x, all_y, all_z = [], [], []
    all_vols = []
    all_stage, all_xmom, all_ymom = [], [], []
    
    point_offset = 0
    with Dataset(sww_files[0], 'r') as first:
        times = first.variables['time'][:]
        ntimes = len(times)
        atts = first.__dict__

    for i, f in enumerate(sww_files):
        if verbose and i % 5 == 0: print(f"  Reading file {i}...")
        with Dataset(f, 'r') as nc:
            pts_in_file = len(nc.dimensions['number_of_points'])
            all_x.append(nc.variables['x'][:])
            all_y.append(nc.variables['y'][:])
            all_z.append(nc.variables['elevation'][:])
            all_vols.append(nc.variables['volumes'][:] + point_offset)
            all_stage.append(nc.variables['stage'][:, :])
            all_xmom.append(nc.variables['xmomentum'][:, :])
            all_ymom.append(nc.variables['ymomentum'][:, :])
            point_offset += pts_in_file

    # Concatenate into large arrays (still containing duplicates)
    raw_x = np.concatenate(all_x)
    raw_y = np.concatenate(all_y)
    raw_z = np.concatenate(all_z)
    raw_vols = np.concatenate(all_vols)
    raw_stage = np.concatenate(all_stage, axis=1)
    raw_xmom = np.concatenate(all_xmom, axis=1)
    raw_ymom = np.concatenate(all_ymom, axis=1)

    # 2. WELDING: Use unique coordinates to identify shared nodes
    # We round to 6 decimal places to ensure floating point matches
    coords = np.round(np.column_stack((raw_x, raw_y)), decimals=6)
    _, unique_idx, inverse_idx = np.unique(coords, axis=0, return_index=True, return_inverse=True)
    
    clean_x = raw_x[unique_idx]
    clean_y = raw_y[unique_idx]
    clean_z = raw_z[unique_idx]
    
    # 3. RE-MAP: Triangles and States to use unique Node IDs
    clean_vols = inverse_idx[raw_vols]
    clean_stage = raw_stage[:, unique_idx]
    clean_xmom = raw_xmom[:, unique_idx]
    clean_ymom = raw_ymom[:, unique_idx]

    # 4. WRITE MERGED FILE
    merged_path = os.path.join(directory, output_name)
    with Dataset(merged_path, 'w', format='NETCDF3_64BIT') as dst:
        dst.createDimension('number_of_volumes', len(clean_vols))
        dst.createDimension('number_of_vertices', 3)
        dst.createDimension('number_of_points', len(clean_x))
        dst.createDimension('number_of_timesteps', ntimes)
        
        dst.setncatts(atts)
        
        dst.createVariable('x', 'f8', ('number_of_points',))[:] = clean_x
        dst.createVariable('y', 'f8', ('number_of_points',))[:] = clean_y
        dst.createVariable('elevation', 'f8', ('number_of_points',))[:] = clean_z
        dst.createVariable('volumes', 'i4', ('number_of_volumes', 'number_of_vertices'))[:] = clean_vols
        dst.createVariable('time', 'f8', ('number_of_timesteps',))[:] = times
        
        dst.createVariable('stage', 'f8', ('number_of_timesteps', 'number_of_points'))[:,:] = clean_stage
        dst.createVariable('xmomentum', 'f8', ('number_of_timesteps', 'number_of_points'))[:,:] = clean_xmom
        dst.createVariable('ymomentum', 'f8', ('number_of_timesteps', 'number_of_points'))[:,:] = clean_ymom

    if verbose:
        print(f"✓ Merge Complete: {len(clean_x)} unique nodes identified (from {len(raw_x)}).")

    # 5. OPTIONAL: Delete originals
    if delete_originals:
        if verbose: print(f"Cleaning up {len(sww_files)} partial files...")
        for f in sww_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"  Error deleting {f}: {e}")
        if verbose: print("✓ Cleanup finished.")

    return merged_path




def merge_sww_files_parallel_parts(directory='results', output_name='merged.sww', 
                    delete_originals=False, verbose=False):
    """
    Merge all parallel ANUGA SWW files in a directory.
    
    Parameters:
    -----------
    directory : str
        Directory containing the partial SWW files
    output_name : str
        Name for the merged output file (will be placed in directory)
    delete_originals : bool
        If True, delete original partial files after successful merge
    verbose : bool
        Print progress messages
    
    Returns:
    --------
    str : Path to the merged file
    """
    
    # ========================================================================
    # Find all SWW files in directory
    # ========================================================================
    pattern = os.path.join(directory, '*_P*.sww')
    sww_files = sorted(glob.glob(pattern))
    
    if len(sww_files) == 0:
        # Try alternate pattern
        pattern = os.path.join(directory, '*.sww')
        sww_files = sorted(glob.glob(pattern))
        
        # Exclude already merged files
        sww_files = [f for f in sww_files if 'merged' not in f.lower() 
                     and 'MERGED' not in f]
    
    if len(sww_files) == 0:
        raise FileNotFoundError(f"No SWW files found in {directory}/")
    
    if verbose:
        print("=" * 60)
        print("MERGING PARALLEL ANUGA SWW FILES")
        print("=" * 60)
        print(f"\nFound {len(sww_files)} partial files in {directory}/:")
        total_size = 0
        for f in sww_files:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            total_size += size_mb
            print(f"  {os.path.basename(f)} ({size_mb:.1f} MB)")
        print(f"Total input size: {total_size:.1f} MB")
    
    # ========================================================================
    # Merge files
    # ========================================================================
    merged_path = os.path.join(directory, output_name)
    
    if verbose:
        print(f"\nMerging into: {merged_path}")
    
    # Open first file to get structure
    with Dataset(sww_files[0], 'r') as src:
        # Get time array
        times = src.variables['time'][:]
        ntimes = len(times)
        
        # Calculate merged dimensions
        total_points = sum(len(Dataset(f, 'r').dimensions['number_of_points']) 
                          for f in sww_files)
        total_volumes = sum(len(Dataset(f, 'r').dimensions['number_of_volumes']) 
                           for f in sww_files)
        
        if verbose:
            print(f"Time steps: {ntimes}")
            print(f"Total points: {total_points:,}")
            print(f"Total volumes: {total_volumes:,}")
        
        # Create merged file
        with Dataset(merged_path, 'w', format='NETCDF3_64BIT') as dst:
            # Create dimensions with merged sizes
            dst.createDimension('number_of_volumes', total_volumes)
            dst.createDimension('number_of_vertices', 3)
            dst.createDimension('number_of_points', total_points)
            dst.createDimension('number_of_timesteps', ntimes)
            
            # Copy global attributes
            dst.setncatts(src.__dict__)
            
            # Create variables
            if verbose:
                print("Creating variables...")
            
            # Static geometry
            x_var = dst.createVariable('x', 'f8', ('number_of_points',))
            y_var = dst.createVariable('y', 'f8', ('number_of_points',))
            z_var = dst.createVariable('elevation', 'f8', ('number_of_points',))
            volumes_var = dst.createVariable('volumes', 'i4', 
                                            ('number_of_volumes', 'number_of_vertices'))
            
            # Time
            time_var = dst.createVariable('time', 'f8', ('number_of_timesteps',))
            time_var[:] = times
            
            # Time-dependent quantities
            stage_var = dst.createVariable('stage', 'f8', 
                                          ('number_of_timesteps', 'number_of_points'))
            xmom_var = dst.createVariable('xmomentum', 'f8', 
                                         ('number_of_timesteps', 'number_of_points'))
            ymom_var = dst.createVariable('ymomentum', 'f8', 
                                         ('number_of_timesteps', 'number_of_points'))
            
            # Collect data from all processors
            if verbose:
                print("Reading and concatenating all data...")
            
            all_x = []
            all_y = []
            all_z = []
            all_volumes = []
            all_stage = []
            all_xmom = []
            all_ymom = []
            
            point_offset = 0
            
            for i, sww_file in enumerate(sww_files):
                if verbose:
                    print(f"  Reading processor {i+1}/{len(sww_files)}...")
                
                with Dataset(sww_file, 'r') as nc:
                    npoints = len(nc.dimensions['number_of_points'])
                    
                    # Read all data at once
                    all_x.append(nc.variables['x'][:])
                    all_y.append(nc.variables['y'][:])
                    all_z.append(nc.variables['elevation'][:])
                    
                    # Adjust volume indices
                    vols = nc.variables['volumes'][:] + point_offset
                    all_volumes.append(vols)
                    
                    # Read all timesteps at once
                    all_stage.append(nc.variables['stage'][:, :])
                    all_xmom.append(nc.variables['xmomentum'][:, :])
                    all_ymom.append(nc.variables['ymomentum'][:, :])
                    
                    point_offset += npoints
            
            if verbose:
                print("Concatenating arrays...")
            
            # Concatenate geometry
            x_var[:] = np.concatenate(all_x)
            y_var[:] = np.concatenate(all_y)
            z_var[:] = np.concatenate(all_z)
            volumes_var[:] = np.concatenate(all_volumes)
            
            # Concatenate time-dependent data along spatial axis
            stage_var[:, :] = np.concatenate(all_stage, axis=1)
            xmom_var[:, :] = np.concatenate(all_xmom, axis=1)
            ymom_var[:, :] = np.concatenate(all_ymom, axis=1)
    
    # ========================================================================
    # Verify merge was successful
    # ========================================================================
    if not os.path.exists(merged_path):
        raise IOError(f"Merge failed - output file not created: {merged_path}")
    
    merged_size_mb = os.path.getsize(merged_path) / (1024 * 1024)
    
    # ========================================================================
    # Delete original files if requested
    # ========================================================================
    if delete_originals:
        if verbose:
            print("\nDeleting original partial files...")
        
        deleted_count = 0
        for sww_file in sww_files:
            try:
                os.remove(sww_file)
                if verbose:
                    print(f"  Deleted: {os.path.basename(sww_file)}")
                deleted_count += 1
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not delete {os.path.basename(sww_file)}: {e}")
        
        if verbose:
            print(f"\nDeleted {deleted_count}/{len(sww_files)} original files")
    
    # ========================================================================
    # Report results
    # ========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("MERGE COMPLETE!")
        print("=" * 60)
        print(f"Merged file: {merged_path}")
        print(f"File size: {merged_size_mb:.1f} MB")
        
        if delete_originals:
            print(f"\n✓ Original {len(sww_files)} partial files deleted")
            print(f"  Disk space saved: ~{total_size - merged_size_mb:.1f} MB")
        else:
            print(f"\nOriginal {len(sww_files)} partial files preserved")
        
        print("=" * 60)
    
    return merged_path
