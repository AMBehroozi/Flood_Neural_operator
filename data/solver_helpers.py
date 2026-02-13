import os
import sys
import glob
from netCDF4 import Dataset
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt



# ==========================================
# Method 1: Rate of Change Analysis
# ==========================================
def check_steady_state_rate_of_change(depth_all, time_array, threshold=1e-4):
    """
    Check steady state by analyzing rate of change over time
    """
    # Calculate time differences
    dt = np.diff(time_array)
    
    # Calculate rate of change for each node
    depth_rate = np.diff(depth_all, axis=0) / dt[:, np.newaxis]
    
    # Calculate global rate of change (RMS across all nodes)
    global_rate = np.sqrt(np.mean(depth_rate**2, axis=1))
    
    # Check if rate is below threshold
    steady_state_reached = global_rate[-10:].mean() < threshold
    
    return global_rate, steady_state_reached

# ==========================================
# Method 2: Maximum Depth Change Between Timesteps
# ==========================================
def check_steady_state_max_change(depth_all, time_array, threshold=0.001):
    """
    Check steady state by maximum depth change between consecutive timesteps
    """
    # Maximum absolute change at any node between timesteps
    max_changes = np.max(np.abs(np.diff(depth_all, axis=0)), axis=1)
    
    # Check if recent changes are below threshold
    steady_state_reached = max_changes[-10:].mean() < threshold
    
    return max_changes, steady_state_reached

# ==========================================
# Method 3: Total Volume Conservation
# ==========================================
def check_steady_state_volume(depth_all, volumes, x, y, time_array):
    """
    Check steady state by tracking total water volume over time
    """
    # Calculate area of each triangle
    x_tri = x[volumes]
    y_tri = y[volumes]
    areas = 0.5 * np.abs(
        (x_tri[:, 1] - x_tri[:, 0]) * (y_tri[:, 2] - y_tri[:, 0]) -
        (x_tri[:, 2] - x_tri[:, 0]) * (y_tri[:, 1] - y_tri[:, 0])
    )
    
    # Calculate total volume at each timestep
    total_volumes = []
    for i in range(len(time_array)):
        # Average depth in each triangle
        avg_depth_per_triangle = depth_all[i, volumes].mean(axis=1)
        total_vol = np.sum(avg_depth_per_triangle * areas)
        total_volumes.append(total_vol)
    
    total_volumes = np.array(total_volumes)
    
    # Rate of volume change
    volume_rate = np.abs(np.diff(total_volumes))
    
    return total_volumes, volume_rate

# ==========================================
# Method 4: Statistical Convergence
# ==========================================
def check_steady_state_statistics(depth_all, window=50, threshold=0.01):
    """
    Check if statistical properties (mean, std) stabilize
    """
    n_timesteps = depth_all.shape[0]
    
    mean_depths = np.mean(depth_all, axis=1)
    std_depths = np.std(depth_all, axis=1)
    
    # Check if mean and std are stable in recent window
    if n_timesteps > window:
        recent_mean_change = np.std(mean_depths[-window:]) / np.mean(mean_depths[-window:])
        recent_std_change = np.std(std_depths[-window:]) / np.mean(std_depths[-window:])
        
        steady_state_reached = (recent_mean_change < threshold) and (recent_std_change < threshold)
    else:
        steady_state_reached = False
    
    return mean_depths, std_depths, steady_state_reached



def loading_data(sww_file, min_depth_threshold=0.01):
    """
    Loads ANUGA .sww file, calculates water depth, removes NaNs, 
    and applies a wet/dry threshold.
    """
    print(f"Loading data from: {sww_file}")
    ds = Dataset(sww_file, 'r')

    # 1. Extract static mesh data
    x = ds.variables['x'][:]
    y = ds.variables['y'][:]
    volumes = ds.variables['volumes'][:]     # Triangle connectivity
    elevation = ds.variables['elevation'][:] # Shape: (num_points,)

    # 2. Extract dynamic data (Stage)
    # Shape: (time, num_points)
    stage_all = ds.variables['stage'][:] 
    time_array = ds.variables['time'][:]
    
    ds.close() # Close file after loading data into memory

    # 3. Calculate Depth
    # NumPy broadcasts (time, points) - (points,) automatically.
    depth_all = stage_all - elevation

    # 4. Clean and Threshold
    # Step A: Clean existing NaNs (Method 1: Replace NaN with 0.0)
    # This fixes the "nan, nan" min/max issue
    depth_all = np.nan_to_num(depth_all, nan=0.0)

    # Step B: Enforce Threshold (Hard reset shallow water to 0.0)
    depth_all[depth_all < min_depth_threshold] = 0.0

    # 5. Verification Prints
    print(f"Depth calculated with threshold {min_depth_threshold}m.")
    print("Shape of depth_all:", depth_all.shape)
    print("Final Min Depth:", depth_all.min())
    print("Final Max Depth:", depth_all.max())
    return x, y, volumes, elevation, depth_all, time_array



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





def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _value_noise_2d(shape, grid_shape, rng):
    """
    Perlin-like *value noise*:
    - Create a coarse random grid
    - Bilinearly interpolate to full resolution
    - Use smoothstep for smoother gradients
    """
    H, W = shape
    gh, gw = grid_shape

    # Random values on coarse lattice
    lattice = rng.uniform(-1.0, 1.0, size=(gh + 1, gw + 1))

    # Coordinates in coarse grid space
    ys = np.linspace(0, gh, H, endpoint=False)
    xs = np.linspace(0, gw, W, endpoint=False)

    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = y0 + 1
    x1 = x0 + 1

    fy = ys - y0
    fx = xs - x0
    fy = _smoothstep(fy)
    fx = _smoothstep(fx)

    # Broadcast to 2D
    y0 = y0[:, None]
    y1 = y1[:, None]
    x0 = x0[None, :]
    x1 = x1[None, :]

    fy = fy[:, None]
    fx = fx[None, :]

    v00 = lattice[y0, x0]
    v10 = lattice[y1, x0]
    v01 = lattice[y0, x1]
    v11 = lattice[y1, x1]

    vx0 = v00 * (1 - fx) + v01 * fx
    vx1 = v10 * (1 - fx) + v11 * fx
    vxy = vx0 * (1 - fy) + vx1 * fy
    return vxy


def _fractal_noise_2d(shape, base_grid=(8, 8), octaves=4, lacunarity=2.0, persistence=0.5, seed=0):
    """
    Multi-octave coherent noise (fractal / fBm style).
    """
    rng = np.random.default_rng(seed)
    H, W = shape

    noise = np.zeros((H, W), dtype=np.float32)
    amp = 1.0
    freq_h, freq_w = base_grid

    amp_sum = 0.0
    for _ in range(octaves):
        n = _value_noise_2d((H, W), (int(freq_h), int(freq_w)), rng)
        noise += amp * n.astype(np.float32)
        amp_sum += amp

        amp *= persistence
        freq_h *= lacunarity
        freq_w *= lacunarity

    noise /= max(amp_sum, 1e-8)
    return noise


def add_coherent_noise_to_dem_ascii(
    input_dem_path,
    output_dem_path,
    amplitude_m=0.5,
    base_grid=(8, 8),
    octaves=4,
    lacunarity=2.0,
    persistence=0.5,
    seed=42,
    mode="std"  # "std" or "max"
):
    """
    Add spatially coherent multi-scale noise to ESRI ASCII DEM and save.

    amplitude_m:
      - if mode="std": scales noise to have std = amplitude_m (meters)
      - if mode="max": scales noise so max(|noise|) = amplitude_m (meters)
    """
    # Read DEM
    with open(input_dem_path, "r") as f:
        header = [next(f) for _ in range(6)]
        data = np.loadtxt(f)

    # Parse NODATA
    nodata_value = None
    for line in header:
        if "NODATA_value" in line or "nodata_value" in line:
            nodata_value = float(line.split()[-1])
            break
    if nodata_value is None:
        raise ValueError("No NODATA_value found in header.")

    mask = data != nodata_value

    # Generate coherent noise field
    n = _fractal_noise_2d(data.shape, base_grid=base_grid, octaves=octaves,
                          lacunarity=lacunarity, persistence=persistence, seed=seed)

    # Scale noise
    if mode == "std":
        s = float(n[mask].std()) if mask.any() else float(n.std())
        n = n * (amplitude_m / max(s, 1e-8))
    elif mode == "max":
        m = float(np.max(np.abs(n[mask]))) if mask.any() else float(np.max(np.abs(n)))
        n = n * (amplitude_m / max(m, 1e-8))
    else:
        raise ValueError('mode must be "std" or "max"')

    # Apply only to valid cells
    noisy = data.copy()
    noisy[mask] = data[mask] + n[mask]

    # Write output
    with open(output_dem_path, "w") as f:
        for line in header:
            f.write(line)
        np.savetxt(f, noisy, fmt="%.4f")

    return output_dem_path

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
