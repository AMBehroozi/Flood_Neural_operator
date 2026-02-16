#!/bin/bash
#SBATCH --job-name=chowilla_river_Grp_%a
#SBATCH --account=cxs1024_cr_default
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --array=0-49
#SBATCH --output=../slurm_log/chowilla_river_grp_%a_%j.out
#SBATCH --error=../slurm_log/chowilla_river_grp_%a_%j.err

module purge
module load anaconda3
module load openmpi/4.1.1-pmi2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MHPI_FLOOD

cd /storage/work/amb10399/project/MHPI_FLOOD/Hurricane_Matthew_Flood/ || exit 1

# Convert the Task ID (0,1,2...) into group_000, group_001, ...
printf -v GROUP_NAME "group_%03d" "$SLURM_ARRAY_TASK_ID"

echo "=========================================="
echo "ARRAY JOB STARTING"
echo "Group Name:  $GROUP_NAME"
echo "Array ID:    $SLURM_ARRAY_TASK_ID"
echo "Job ID:      $SLURM_JOB_ID"
echo "Tasks:       $SLURM_NTASKS"
echo "Date:        $(date)"
echo "Using mpirun: $(which mpirun)"
mpirun --version | head -n 3
echo "=========================================="

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Run MPI parallel job (same style as your reference script)
mpirun -np "$SLURM_NTASKS" python data/chowilla_river/chowilla_river_simulation.py "$GROUP_NAME"

echo "Job $GROUP_NAME completed: $(date)"
