#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --partition=standard
#SBATCH --account=cxs1024_cr_default
#SBATCH --job-name=Hurricane_run
#SBATCH --output=../../slurm_log/slurm-%j.out
#SBATCH --error=../../slurm_log/slurm-%j.err

module purge
module load anaconda3

# Robust conda init in batch jobs
source $(conda info --base)/etc/profile.d/conda.sh
conda activate MHPI_FLOOD

cd /storage/work/amb10399/project/MHPI_FLOOD/Hurricane_Matthew_Flood/data/chowilla_river

echo "=========================================="
echo "JOB STARTING"
echo "Job ID:   $SLURM_JOB_ID"
echo "Tasks:    $SLURM_NTASKS"
echo "Date:     $(date)"
echo "Python:   $(which python)"
echo "=========================================="

# Launch 40 MPI ranks
srun -n $SLURM_NTASKS python reaching_steady_state_parallel.py

echo "Job completed: $(date)"
