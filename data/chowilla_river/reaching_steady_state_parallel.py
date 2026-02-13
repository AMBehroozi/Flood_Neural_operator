import os
import sys
import os
import glob
from netCDF4 import Dataset
import numpy as np
import pandas as pd

# ============================================================================
# FORCE MPI INITIALIZATION FIRST
# ============================================================================
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    numprocs = comm.Get_size()
    print(f"MPI initialized: Processor {myid} of {numprocs}", flush=True)
except ImportError:
    print("WARNING: mpi4py not available — running in serial mode")
    myid = 0
    numprocs = 1

# Import ANUGA parallel utilities
import anuga
from anuga.parallel import distribute, barrier, finalize

# ============================================================================
# Configuration
# ============================================================================
topography_file = '/storage/work/amb10399/project/MHPI_FLOOD/Hurricane_Matthew_Flood/data/chowilla_river/DEM_data/chowilla_dem_final_single.asc'
polyline_csv='/storage/work/amb10399/project/MHPI_FLOOD/Hurricane_Matthew_Flood/data/chowilla_river/DEM_data/chowilla_finer_area.csv'
points_csv='/storage/work/amb10399/project/MHPI_FLOOD/Hurricane_Matthew_Flood/data/chowilla_river/DEM_data/chowilla_BCs.csv'
mesh_file = '/storage/work/amb10399/project/MHPI_FLOOD/Hurricane_Matthew_Flood/data/chowilla_river/DEM_data/Chowilla_River_mesh_fine.msh'  # Pre-generated mesh

# merge_sww.py
# Function to merge parallel ANUGA SWW files



# ============================================================================
# Parallel status report
# ============================================================================
if myid == 0:
    print("=" * 60)
    print(f"PARALLEL ANUGA SIMULATION — Using {numprocs} processor(s)")
    print("=" * 60)
    if numprocs == 1:
        print("\n⚠️  Running in SERIAL mode (no parallel speedup)\n")

barrier()
case_path = '/storage/work/amb10399/project/MHPI_FLOOD/Hurricane_Matthew_Flood/data/chowilla_river/results'
case_name = 'chowilla'
# ============================================================================
# MASTER (rank 0): Load mesh and configure full domain
# ============================================================================
if myid == 0:
    print(f"\nLoading mesh from: {mesh_file}")
    domain = anuga.Domain(mesh_file, use_cache=False, verbose=True)

    domain.set_name(case_name)
    domain.set_datadir(case_path)

    print(f"Mesh: {domain.number_of_triangles} triangles")
    print(f"Approx. triangles per processor: {domain.number_of_triangles // max(numprocs, 1)}")





    # Boundary conditions
    print("\nSetting boundary conditions...")
    wall_BC = anuga.Reflective_boundary(domain)
    # outflow_BC = anuga.Transmissive_boundary(domain)
    outflow_BC = anuga.Dirichlet_boundary([10.0, 0.0, 0.0])


    # Apply outflow to ALL exterior boundaries
    domain.set_boundary({
        'south': outflow_BC,          # ← change 'bc' to your actual boundary tag name
        'north': wall_BC,
        'west' : wall_BC,
        'east' : wall_BC
        # or 'boundary': outflow_BC,
    })

    # Initial conditions
    print("Setting initial conditions...")
    domain.set_quantity('elevation', filename=topography_file, location='centroids')
    domain.set_quantity('friction', 0.06, location='centroids')
    domain.set_quantity('stage', expression='elevation', location='centroids')
    domain.set_quantity('xmomentum', 0.0, location='centroids')
    domain.set_quantity('ymomentum', 0.0, location='centroids')




    domain.set_minimum_storable_height(0.025)

    print("Master setup complete.\n")
else:
    domain = None

# ============================================================================
# DISTRIBUTE DOMAIN TO ALL PROCESSORS
# ============================================================================
domain = distribute(domain)

if myid == 0:
    print("Domain successfully distributed to all processors.\n")

# Synchronize before adding forcing terms
barrier()

# ============================================================================
# ALL PROCESSORS: Add inlet operators (MUST be created on every rank)
# ============================================================================
if myid == 0:
    print("Creating inlet operators on all processors...")


# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
Q_CONST = [71.0, 11.0, 1.0]          # m3/s per inlet
N_INLETS = 3             # ONLY first 3 points
SAFETY = 1.2             # 20% margin on radius


# ────────────────────────────────────────────────
# LOAD FIRST 3 POINTS ONLY
# ────────────────────────────────────────────────
points_df = pd.read_csv(
    points_csv,
    header=None,
    names=["x", "y"]
).iloc[:N_INLETS]

print(f"Using {len(points_df)} inlet point(s) (first {N_INLETS} only)")


# ────────────────────────────────────────────────
# PRECOMPUTE CENTROIDS ONCE
# ────────────────────────────────────────────────
centroids = domain.get_centroid_coordinates()


# ────────────────────────────────────────────────
# ASSIGN INLETS (AUTO RADIUS)
# ────────────────────────────────────────────────
inlets = []

for i, row in points_df.iterrows():
    center = (float(row["x"]), float(row["y"]))
    c = np.array(center)

    # Distance to nearest centroid
    d = np.sqrt(((centroids - c) ** 2).sum(axis=1))
    radius_used = d.min() * SAFETY

    region = anuga.Region(
        domain,
        center=center,
        radius=150
    )

    inlet = anuga.Inlet_operator(
        domain,
        region,
        Q=Q_CONST[i]
    )

    inlets.append(inlet)

if myid == 0:
    print(
        f"\nSuccess: {len(inlets)} inlets created "
        f"(first {N_INLETS} CSV points, Q = {Q_CONST} m³/s each)"
    )



if myid == 0:
    print("All inlets configured.\n")

# ============================================================================
# Set quantities to be stored
# ============================================================================
domain.set_quantities_to_be_stored({
    'stage': 2,
    'xmomentum': 2,
    'ymomentum': 2,
    'elevation': 1,
    'friction': 1
})

# ============================================================================
# CRITICAL: Initialize SWW output files by storing t=0
# ============================================================================
domain.set_store(True)
# domain.store_timestep()  # Prevents KeyError: 'time' in parallel

if myid == 0:
    print("SWW files initialized with initial timestep.\n")


import logging

# ============================================================================
# RUN SIMULATION
# ============================================================================
DAY = 24 * 3600  # Fixed: full day in seconds

if myid == 0:
    log_dir = "../../logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "simulation_progress.log")

    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    logging.info("=" * 60)
    logging.info("STARTING 20-DAY SIMULATION")
    logging.info(f"Yieldstep: 10 days")
    logging.info("=" * 60)

for t in domain.evolve(yieldstep=2 * DAY, finaltime=200 * DAY):
    if myid == 0:
        logging.info(f"Current time: {t / DAY:.2f} days")
        domain.print_timestepping_statistics()

if myid == 0:
    logging.info("SIMULATION COMPLETED")
    logging.info("=" * 60)

# Clean parallel shutdown
finalize()
