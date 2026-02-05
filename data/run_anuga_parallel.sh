#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --partition=standard
#SBATCH --account=cxs1024_cr_default
#SBATCH --array=0-19
#SBATCH --job-name=Hurricane_Grp_%a
#SBATCH --output=logs/hurricane_grp_%a_%j.out
#SBATCH --error=logs/hurricane_grp_%a_%j.err

module purge
module load anaconda3

# Initialize conda for this script
source $(conda info --base)/etc/profile.d/conda.sh
conda activate MHPI_FLOOD

cd /storage/work/amb10399/project/MHPI_FLOOD/Hurricane_Matthew_Flood/

# Create logs directory if it doesn't exist
mkdir -p Slurm_logs

# Convert the Task ID into group format
printf -v GROUP_NAME "group_%03d" $SLURM_ARRAY_TASK_ID

echo "=========================================="
echo "ARRAY JOB STARTING"
echo "Group Name:  $GROUP_NAME"
echo "Array ID:    $SLURM_ARRAY_TASK_ID"
echo "Job ID:      $SLURM_JOB_ID"
echo "Tasks:       $SLURM_NTASKS"
echo "Date:        $(date)"
echo "=========================================="

# ANUGA handles its own parallelization - just run with python
python data/dam_break/dam_break_simulation_UQ.py "$GROUP_NAME"

echo "Job $GROUP_NAME completed: $(date)"