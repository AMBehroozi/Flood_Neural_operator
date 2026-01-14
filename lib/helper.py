import os
import sys
import glob
from netCDF4 import Dataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

def coarsen_spatial_tensor(tensor, N):
    """
    Coarsens a tensor of shape [nb, nx, ny, nt] by a factor of N 
    using spatial averaging. If N=1, returns the original tensor.
    """
    # 0. Early exit for no coarsening
    if N == 1:
        return tensor
        
    # 1. Capture original shapes
    nb, nx, ny, nt = tensor.shape
    
    # 2. Reshape to [nb * nt, 1, nx, ny] 
    # Move nt into the batch dimension for 2D pooling
    x = tensor.permute(0, 3, 1, 2).reshape(nb * nt, 1, nx, ny)
    
    # 3. Apply Average Pooling
    # kernel_size and stride are both N
    x_coarse = F.avg_pool2d(x, kernel_size=N, stride=N)
    
    # 4. Get new spatial dimensions
    _, _, nx_new, ny_new = x_coarse.shape
    
    # 5. Reshape and Permute back to [nb, nx_new, ny_new, nt]
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
