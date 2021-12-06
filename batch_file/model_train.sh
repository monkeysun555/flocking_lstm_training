#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=50:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=model_train
#SBATCH --mail-type=END
#SBATCH --mail-user=ls3817@nyu.edu
#SBATCH --output=slurm_%j.out
  
module purge

VIRTUALENV=$SCRATCH
RUNDIR=$SCRATCH/live_360/enhanced_live_360/fov_prediction

cd $VIRTUALENV

source ./keras/bin/activate
  
cd $RUNDIR
python convLSTM_heatmap.py
