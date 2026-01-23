from mpi4py import MPI
import sys
import os

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()

    # Print from every rank
    print(f"Hello from Rank {rank}/{size} on host {name}")

    # Synchronize
    comm.Barrier()

    if rank == 0:
        print(f"MPI Test Successful! Python version: {sys.version}")
        print(f"Working Directory: {os.getcwd()}")

if __name__ == "__main__":
    main()