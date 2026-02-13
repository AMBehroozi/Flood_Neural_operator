#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=24:00:00
#SBATCH --job-name='chowilla_flooding_simulation'
#SBATCH --partition=mgc-mri
#SBATCH --account=cxs1024_mri
#SBATCH --output=../../slurm_log/slurm-%j.out
#SBATCH --error=../../slurm_log/slurm-%j.err

echo "Job started: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Running on node(s): $SLURM_NODELIST"

module purge
module load anaconda3
source activate MHPI_FLOOD


# ────────────────────────────────────────────────
# Papermill execution (EXPLICIT output notebook)
# ────────────────────────────────────────────────
INPUT_NOTEBOOK="chowilla_flooding_simulation.ipynb"
OUTPUT_NOTEBOOK="../../slurm_log/chowilla_flooding_simulation_out.ipynb"

papermill "$INPUT_NOTEBOOK" "$OUTPUT_NOTEBOOK"
mpirun -np 20 --oversubscribe python reaching_steady_state_parallel.py