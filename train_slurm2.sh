#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --job-name='chowilla_river_config_stage1'
#SBATCH --account=cxs1024_cr_default
#SBATCH --partition=standard
#SBATCH --output=slurm_log/slurm-%j.out
#SBATCH --error=slurm_log/slurm-%j.err

# create slurm log directory
mkdir -p slurm_log

echo "Job started: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"

module purge
module load anaconda3
source activate FNO
cd /storage/work/amb10399/project/MHPI_FLOOD/Hurricane_Matthew_Flood/

python3 FNO_forward/FNO_Trainer/FNO_parallel_trainer_multi_stream.py
