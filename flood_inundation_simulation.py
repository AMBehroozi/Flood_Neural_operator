import os
import sys
import glob
from netCDF4 import Dataset
import numpy as np
import anuga
from anuga.parallel import distribute, barrier, finalize
from helpers.helper import *
import rasterio
from scipy.interpolate import griddata

class HurricaneSimulation:
    def __init__(self, myid, numprocs, topography_file, mesh_file, finer_zone_path, 
                 sww_input, sww_continue, base_resolution, finer_zone_resolution, 
                 tms_dir, radius, gauges):
        self.myid = myid
        self.numprocs = numprocs
        self.topography_file = topography_file
        self.mesh_file = mesh_file
        self.finer_zone_path = finer_zone_path
        self.sww_input = sww_input
        self.sww_continue = sww_continue
        self.base_resolution = base_resolution
        self.finer_zone_resolution = finer_zone_resolution
        self.tms_dir = tms_dir
        self.radius = radius
        self.gauges = gauges
        
        # Internal state
        self.domain = None
        self.final_time_seconds = 0.0
        self.DAY = 24 * 3600
        self.threshold = 0.025

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

        self.domain = anuga.create_domain_from_regions(
            bounding_polygon=bounding_polygon,
            boundary_tags={'west': [0], 'north': [1], 'east': [2], 'south': [3]},
            maximum_triangle_area=self.base_resolution,
            interior_regions=interior_regions,
            mesh_filename=self.mesh_file,
            use_cache=False,
            verbose=False
        )
        self.domain.set_name(self.sww_continue)

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
        outflow_BC = anuga.Dirichlet_boundary([-10, 0.0, 0.0]) 
        self.domain.set_boundary({'south': wall_BC, 'east': outflow_BC, 
                                  'north': outflow_BC, 'west': wall_BC})

        self.domain.set_starttime(0.0)
        self.domain.set_minimum_allowed_height(self.threshold)
        self.domain.set_store(True)
        self.domain.set_quantities_to_be_stored({'stage': 2, 'xmomentum': 2, 'ymomentum': 2, 'elevation': 1, 'friction': 1})

    def setup_inlets(self):
        """Sets up inlet operators and calculates total simulation time."""
        max_time = 0.0
        for gauge in self.gauges:
            usgs_id = gauge['id'].split('_')[-1]
            tms_filename = os.path.join(self.tms_dir, f"{usgs_id}.tms")
            
            if os.path.exists(tms_filename):
                with Dataset(tms_filename, 'r') as nc:
                    file_end_time = nc.variables['time'][-1]
                    max_time = max(max_time, file_end_time)
                
                Q_func = anuga.file_function(tms_filename, quantities='discharge')
                region = anuga.Region(self.domain, center=(gauge['x'], gauge['y']), radius=self.radius)
                anuga.Inlet_operator(self.domain, region, Q=Q_func)
        
        self.final_time_seconds = float(max_time)

    def evolve(self, yieldstep_factor=0.25):
        """Runs the simulation evolution loop."""
        if self.myid == 0:
            print(f"Starting Evolution. Final time: {self.final_time_seconds/3600:.2f} hours")
        
        for t in self.domain.evolve(yieldstep=yieldstep_factor * self.DAY, finaltime=self.final_time_seconds):
        # for t in self.domain.evolve(yieldstep=yieldstep_factor * self.DAY, finaltime=2 * self.DAY):
            if self.myid == 0:
                self.domain.print_timestepping_statistics()
                print(f"   Current Progress: {t / self.DAY:.2f} days")




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
    except ImportError:
        myid = 0
        numprocs = 1

    # Raw Input Variables
    data_dir = '/storage/group/cxs1024/default/mehdi/Hurricane_MatthewData/DEM10/'
    topography_file = os.path.join(data_dir, 'DM10GLDN2.asc')
    mesh_file = 'mesh/hurricane_domain_final_backup.msh'
    finer_zone_path = 'finer_zone.csv'
    sww_input = "Hurricane_steady_state_phase_2.sww"
    case_path = 'results/temp/' 
    base_resolution = 70000.0
    finer_zone_resolution = 7000.0

    TMS_OUTPUT_FOLDER = '/storage/group/cxs1024/default/mehdi/Hurricane_MatthewData/tms_files'
    radius = 100.0  
    gauges = [
        {"id": "Bc1_8791413",  "x": 203142.27, "y": 3922226.29},
        {"id": "Bc2_8790751",  "x": 209767.46, "y": 3915832.46},
        {"id": "Bc3_8790719",  "x": 214258.80, "y": 3920627.66},
        {"id": "Bc4_8790801",  "x": 215750.40, "y": 3914499.44},
        {"id": "Bc5_8790519",  "x": 218575.49, "y": 3920699.28},
        {"id": "Bc6_8790559",  "x": 225078.15, "y": 3921002.98},
        {"id": "Bc7_11235707", "x": 227582.00, "y": 3915030.09},
        {"id": "Bc8_11236643", "x": 231177.77, "y": 3904981.34},
    ]

    # --- START OF 2-CYCLE LOOP ---
    for i in range(1, 2):
        if myid == 0:
            print(f"\n" + "="*60)
            print(f"STARTING CYCLE {i}")
            print("="*60)

        # Unique output name for each cycle to prevent overwriting
        sww_continue = os.path.join(case_path, f'Hurricane_cycle_{i}')

        # Instantiate and Run Simulation (All processors participate)
        sim = HurricaneSimulation(
            myid, numprocs, topography_file, mesh_file, finer_zone_path,
            sww_input, sww_continue, base_resolution, finer_zone_resolution,
            TMS_OUTPUT_FOLDER, radius, gauges
        )
        
        sim.setup_domain()
        sim.setup_inlets()
        sim.evolve()
        
        # CRITICAL: Wait for all ranks to finish writing partial files
        barrier()

        # ============================================================================
        # FINALIZE AND MERGE OUTPUT (MASTER PROCESS ONLY)
        # ============================================================================
        if myid == 0:
            print(f"Master: Merging partial files for Cycle {i}...")
            
            # Use the directory where partial files were saved
            merged = merge_sww_files_parallel_parts(
                directory=case_path,
                output_name=f'merged/Hurricane_dynamics_cycle_{i}.sww',
                delete_originals=True,
                verbose=True
            )
            print(f"Merged file created: {merged}")
        
        # Sync again before starting the next cycle
        barrier()

    # Final cleanup after all cycles complete
    finalize()