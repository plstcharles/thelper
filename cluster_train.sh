#!/bin/bash
#SBATCH --job-name=thelper
#SBATCH --output=sessions/logs/%j.log
#SBATCH --error=sessions/logs/%j.log
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=main
#SBATCH --time=23:00:00

echo    "Arguments: $@"
echo -n "Date:      "; date
echo    "JobId:     $SLURM_JOBID"
echo    "JobName:   $SLURM_JOB_NAME"
echo    "Node:      $HOSTNAME"
echo    "Nodelist:  $SLURM_JOB_NODELIST"

export PYTHONUNBUFFERED=1
module purge
module load cuda/10.0/cudnn/7.6
module load miniconda/3
source $CONDA_ACTIVATE
conda activate thelper

"$CONDA_PREFIX/bin/python" thelper/cli.py -vvv new configs/agrivis-ae.yml sessions/
