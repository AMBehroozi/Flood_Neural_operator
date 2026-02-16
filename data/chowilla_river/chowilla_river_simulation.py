import os
import sys
import glob
from netCDF4 import Dataset
import numpy as np
import anuga
from anuga.parallel import distribute, barrier, finalize
import rasterio
from scipy.interpolate import griddata
import datetime
from tqdm import tqdm
import signal
import traceback
import datetime
import sys
import os
import pandas as pd
sys.path.append("./")
from data.solver_helpers import *



class TimeoutException(Exception):
    """Raised when a scenario times out"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Scenario execution timeout")


class HurricaneSimulation:
    def __init__(self, myid, numprocs, topography_file, finer_zone_path, 
                 sww_input, sww_continue, case_path, base_resolution, finer_zone_resolution, 
                 tms_dir, radius, gauges):
        self.myid = myid
        self.numprocs = numprocs
        self.topography_file = topography_file
        self.finer_zone_path = finer_zone_path
        self.sww_input = sww_input
        self.sww_continue = sww_continue
        self.case_path = case_path
        self.base_resolution = base_resolution
        self.finer_zone_resolution = finer_zone_resolution
        self.tms_dir = tms_dir
        self.radius = radius
        self.gauges = gauges
        
        # Internal state
        self.domain = None
        self.final_time_seconds = 0.0
        self.DAY = 24 * 3600
        self.threshold = 0.005

    def setup_domain(self):
        """Orchestrates the preparation and distribution of the domain."""
        if self.myid == 0:
            self._prepare_master_domain()
        else:
            self.domain = None
        
        # Distribute the domain to all processors
        self.domain = distribute(self.domain)
        self._initialize_local_quantities()
        # self._setup_inlets()

    def _prepare_master_domain(self):
        """Master process logic for mesh generation and interpolation."""
        print("Master: Loading previous state and preparing mesh...")
        states = get_sww_mesh_and_states(self.sww_input)
        
        with rasterio.open(self.topography_file) as src:
            x_min, y_min, x_max, y_max = src.bounds

        bounding_polygon = [[x_min, y_min], [x_min, y_max], 
                            [x_max, y_max], [x_max, y_min]]

        finer_zone = anuga.read_polygon(self.finer_zone_path)
        interior_regions = [[finer_zone, self.finer_zone_resolution]]
        
        unique_mesh_filename = os.path.join(self.case_path, f"mesh_{group}_{senario}.msh")
        self.domain = anuga.create_domain_from_regions(
            bounding_polygon=bounding_polygon,
            boundary_tags={'west': [0], 'north': [1], 'east': [2], 'south': [3]},
            maximum_triangle_area=self.base_resolution,
            interior_regions=interior_regions,
            mesh_filename=unique_mesh_filename,
            use_cache=False,
            verbose=False
        )
        self.domain.set_name(self.sww_continue)
        self.domain.set_datadir(case_path)
        # Spatial interpolation of states to new mesh
        print("Master: Performing spatial interpolation...")
        old_coords = states['vertices']
        old_depth = states['stage'] - states['elevation']
        new_centroids = self.domain.get_centroid_coordinates()

        interp_depth = griddata(old_coords, old_depth, new_centroids, method='nearest')
        interp_xmom = griddata(old_coords, states['xmomentum'], new_centroids, method='nearest')
        interp_ymom = griddata(old_coords, states['ymomentum'], new_centroids, method='nearest')

        self.domain.set_quantity('stage', np.nan_to_num(interp_depth, nan=0.0), location='centroids')
        self.domain.set_quantity('xmomentum', np.nan_to_num(interp_xmom, nan=0.0), location='centroids')
        self.domain.set_quantity('ymomentum', np.nan_to_num(interp_ymom, nan=0.0), location='centroids')

    def _initialize_local_quantities(self):
        """Initializes quantities on each parallel sub-domain."""
        # Elevation must be set locally on all ranks
        self.domain.set_quantity('elevation', filename=self.topography_file, location='centroids')
        actual_elev = self.domain.get_quantity('elevation').get_values(location='centroids')

        # Retrieve distributed values and enforce dry-region masking
        p_depth = self.domain.get_quantity('stage').get_values(location='centroids')
        p_xmom = self.domain.get_quantity('xmomentum').get_values(location='centroids')
        p_ymom = self.domain.get_quantity('ymomentum').get_values(location='centroids')

        is_dry = p_depth < self.threshold
        final_p_stage = np.where(is_dry, actual_elev, actual_elev + p_depth)
        final_p_xmom = np.where(is_dry, 0.0, p_xmom)
        final_p_ymom = np.where(is_dry, 0.0, p_ymom)

        self.domain.set_quantity('stage', final_p_stage, location='centroids')
        self.domain.set_quantity('xmomentum', final_p_xmom, location='centroids')
        self.domain.set_quantity('ymomentum', final_p_ymom, location='centroids')
        self.domain.set_quantity('friction', 0.01, location='centroids')

        # Set Boundaries
        
        wall_BC = anuga.Reflective_boundary(self.domain)
        outflow_BC = anuga.Transmissive_boundary(self.domain)
        self.domain.set_boundary({
                'south': outflow_BC,          # ← change 'bc' to your actual boundary tag name
                'north': outflow_BC,
                'west' : outflow_BC,
                'east' : wall_BC
                # or 'boundary': outflow_BC,
                })

        self.domain.set_starttime(0.0)
        self.domain.set_minimum_allowed_height(self.threshold)
        self.domain.set_store(True)
        self.domain.set_quantities_to_be_stored({'stage': 2, 'xmomentum': 2, 'ymomentum': 2, 'elevation': 1, 'friction': 1})

    def setup_inlets(self):
        """Sets up inlet operators and calculates total simulation time."""
        max_time = 0.0
        for gauge in self.gauges:
            usgs_id = gauge['id']
            tms_filename = os.path.join(self.tms_dir, f"{usgs_id}.tms")
            
            if os.path.exists(tms_filename):
                with Dataset(tms_filename, 'r') as nc:
                    file_end_time = nc.variables['time'][-1]
                    max_time = max(max_time, file_end_time)
                
                Q_func = anuga.file_function(tms_filename, quantities='discharge')
                region = anuga.Region(self.domain, center=(gauge['x'], gauge['y']), radius=self.radius)
                anuga.Inlet_operator(self.domain, region, Q=Q_func)
        
        self.final_time_seconds = float(max_time)


    def evolve(self, log_file_path, yieldstep_factor=0.25):
        """Runs the simulation evolution loop with per-iteration timeout."""
        if self.myid == 0:
            print(f"Starting Evolution. Final time: {self.final_time_seconds/3600:.2f} hours")
        
        iteration_timeout = 900  # 15 minutes per iteration
        
        try:
            for t in self.domain.evolve(yieldstep=yieldstep_factor * self.DAY, finaltime=self.final_time_seconds):
                
                # Reset timeout for EACH iteration (critical!)
                signal.alarm(iteration_timeout)
                
                if self.myid == 0:
                    with open(log_file_path, "a") as log:
                        progress_percent = (t / self.final_time_seconds) * 100
                        now = datetime.datetime.now()
                        log.write(f'time: {t:.2f}s ({t/self.DAY:.2f} days, {progress_percent:.1f}%) -- {now.strftime("%d/%m/%y %H:%M")}\n')
                    self.domain.print_timestepping_statistics()
                    print(f"   Current Progress: {t / self.DAY:.2f} days")
                
        except Exception as e:
            if self.myid == 0:
                print(f"Evolution failed at rank {self.myid}: {str(e)}")
            raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # MPI Initialization
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        myid = comm.Get_rank()
        numprocs = comm.Get_size()
        use_mpi = True
    except ImportError:
        comm = None
        myid = 0
        numprocs = 1
        use_mpi = False

    # Raw Input Variables
    data_dir = '/storage/work/amb10399/project/MHPI_FLOOD/Hurricane_Matthew_Flood/data/chowilla_river'
    topography_file = os.path.join(data_dir, 'DEM_data/chowilla_dem_final_single.asc')
    finer_zone_path = os.path.join(data_dir, 'DEM_data/chowilla_finer_area.csv')
    points_csv =      os.path.join(data_dir, 'DEM_data/chowilla_BCs.csv')
    sww_input =       os.path.join(data_dir, 'results/merged.sww')


    Bc_points = (pd.read_csv(points_csv, header=None, names=["x", "y"]))

    base_resolution = 300_000.0
    finer_zone_resolution = 60_000.0

    radius = 150.0  
    gauges = [
        {"id": "1",  "x": Bc_points['x'][0], "y": Bc_points['y'][0]},
        {"id": "2",  "x": Bc_points['x'][1], "y": Bc_points['y'][1]},
        {"id": "3",  "x": Bc_points['x'][2], "y": Bc_points['y'][2]},
    ]



    # group = 'group_000'  # group_XXX

    # --- COMMAND LINE INPUT FOR GROUP ---
    # sys.argv[0] is the script name, sys.argv[1] is the first argument
    if len(sys.argv) > 1:
        group = sys.argv[1]
    else:
        # # Default fallback if no argument is provided
        group = 'group_004'
        if myid == 0:
            print(f"Warning: No group provided. Using default: {group}")


    

    group_path = '/storage/group/cxs1024/default/mehdi/chowilla_river_scenario_groups/scenario_groups_2/' + group
    senaio_list = get_subfolders(group_path)

    # Define the log file path in the root directory
    log_file_path = f"logs/simulation_progress_{group}.log"
    
    # Track failed scenarios
    failed_scenarios = []
    successful_scenarios = []

    # --- START OF SCENARIO LOOP ---
    for senario in senaio_list:
        # 1. CHECK IF ALREADY PROCESSED/MERGED
        case_path = os.path.join(group_path, senario)
        merged_filename = f'chowilla_river_{group}_{senario}_merged.sww'
        merged_file_path = os.path.join(case_path, merged_filename)

        if os.path.exists(merged_file_path):
            if myid == 0:
                print(f"Skipping {senario}: Merged file already exists.")
                # Write to log file immediately
                with open(log_file_path, "a") as log:
                    log.write(f"Skipping {senario}: Merged file already exists.\n")

            continue
        
        
        # Flag to track if this rank encountered an error (0=success, 1=error) domain.set_datadir
        error_occurred = 0
        
        try:
            TMS_OUTPUT_FOLDER = os.path.join(group_path, senario, 'tms_files')
            case_path = os.path.join(group_path, senario)

            # Unique output name for each cycle
            sww_continue = f'chowilla_river_{group}_{senario}'
            if myid == 0:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                start_msg = f"[{timestamp}] STARTING: {group} - {senario}\n"
                print(f"\n" + "="*60 + f"\n{start_msg}\n" + "="*60)
                
                # Write to log file immediately
                with open(log_file_path, "a") as log:
                    log.write(start_msg + "\n")

            # Instantiate and Run Simulation
            sim = HurricaneSimulation(
                myid, numprocs, topography_file, finer_zone_path,
                sww_input, sww_continue, case_path, base_resolution, finer_zone_resolution,
                TMS_OUTPUT_FOLDER, radius, gauges
            )
            
            sim.setup_domain()
            sim.setup_inlets()
            sim.evolve(log_file_path=log_file_path, yieldstep_factor=3)
            
        except Exception as e:
            # Mark that this rank had an error
            error_occurred = 1
            
            # Log error details (all ranks can log to see which rank failed)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_msg = f"[{timestamp}] ERROR in rank {myid} for {group} - {senario}: {str(e)}"
            print(f"\n{'!'*60}\nRANK {myid}: {error_msg}\n{'!'*60}", flush=True)
            
            if myid == 0:
                # Master also logs to file
                with open(log_file_path, "a") as log:
                    log.write(error_msg + "\n")
                    log.write(f"Traceback: {traceback.format_exc()}\n\n")
        
        # Synchronize and check if ANY rank had an error
        try:
            if use_mpi and numprocs > 1:
                # Gather error flags from all ranks to rank 0
                all_errors = comm.gather(error_occurred, root=0)
                
                # Rank 0 determines if any rank failed
                if myid == 0:
                    any_error = sum(all_errors) > 0
                    if any_error:
                        failed_ranks = [i for i, err in enumerate(all_errors) if err > 0]
                        print(f"ERROR detected on rank(s): {failed_ranks}")
                else:
                    any_error = None
                
                # Broadcast the decision to all ranks
                any_error = comm.bcast(any_error, root=0)
            else:
                any_error = error_occurred > 0
        except Exception as e:
            # If communication itself fails, assume error and continue
            print(f"Rank {myid}: Communication error during error check: {e}", flush=True)
            any_error = True
        
        # All ranks now know if the scenario failed
        if any_error:
            if myid == 0:
                print(f"Scenario {senario} failed on one or more ranks. Skipping to next scenario.")
                failed_scenarios.append(senario)
            
            # All ranks skip merging and continue to next scenario
            if use_mpi and numprocs > 1:
                try:
                    comm.Barrier()
                except:
                    pass  # If barrier fails, continue anyway
            continue
        
        # If we reach here, all ranks succeeded
        if use_mpi and numprocs > 1:
            comm.Barrier()
        
        # Master process merges results and updates log
        if myid == 0:
            print(f"Master: Merging partial files for {group}_{senario}...")
            try:
                merge_sww_files_parallel_parts(
                    directory=case_path,
                    output_name=f'chowilla_river_{group}_{senario}_merged.sww',
                    delete_originals=True,
                    verbose=True
                )
                
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                end_msg = f"[{timestamp}] COMPLETED & MERGED: {group} - {senario}"
                print(end_msg)
                
                # Append completion status to log
                with open(log_file_path, "a") as log:
                    log.write(end_msg + "\n\n")
                
                successful_scenarios.append(senario)
            except Exception as e:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                error_msg = f"[{timestamp}] ERROR during merge for {group} - {senario}: {str(e)}"
                print(error_msg)
                with open(log_file_path, "a") as log:
                    log.write(error_msg + "\n\n")
                failed_scenarios.append(senario)
        
        # Final sync for current cycle before moving to the next
        if use_mpi and numprocs > 1:
            comm.Barrier()

    # Final summary
    if myid == 0:
        print("\n" + "="*60)
        print("ALL CYCLES COMPLETE")
        print("="*60)
        print(f"Successful scenarios: {len(successful_scenarios)}/{len(senaio_list)}")
        print(f"Failed scenarios: {len(failed_scenarios)}/{len(senaio_list)}")
        
        if failed_scenarios:
            print("\nFailed scenarios:")
            for fs in failed_scenarios:
                print(f"  - {fs}")
        
        # Write final summary to log
        with open(log_file_path, "a") as log:
            log.write("="*60 + "\n")
            log.write(f"FINAL SUMMARY - {group}\n")
            log.write(f"Successful: {len(successful_scenarios)}/{len(senaio_list)}\n")
            log.write(f"Failed: {len(failed_scenarios)}/{len(senaio_list)}\n")
            if failed_scenarios:
                log.write("Failed scenarios: " + ", ".join(failed_scenarios) + "\n")
            log.write("="*60 + "\n")
        
        print("="*60)
    
    # Final barrier and cleanup
    if use_mpi and numprocs > 1:
        comm.Barrier()
        MPI.Finalize()
    
    sys.exit(0)

