#!/bin/bash
#SBATCH --job-name=chowilla-mpi
#SBATCH --account=cxs1024_cr_default
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=80GB
#SBATCH --time=10:00:00
#SBATCH --output=../../slurm_log/chowilla_%j.out
#SBATCH --error=../../slurm_log/chowilla_%j.err

module purge
module load anaconda3
module load openmpi/4.1.1-pmi2

source $(conda info --base)/etc/profile.d/conda.sh
conda activate MHPI_FLOOD

cd /storage/work/amb10399/project/MHPI_FLOOD/Hurricane_Matthew_Flood/data/chowilla_river || exit 1

echo "Starting job $(date)"
echo "Using mpirun: $(which mpirun)"
mpirun --version | head -n 3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mpirun -np $SLURM_NTASKS python reaching_steady_state_parallel.py \
    > sim_log.txt 2>&1

echo "Finished $(date)"