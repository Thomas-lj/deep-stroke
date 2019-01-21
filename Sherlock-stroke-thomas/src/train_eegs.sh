#!/bin/bash
#SBATCH --job-name=eegs-50epoch
#SBATCH --nodes=1
#SBATCH --error=/scratch/users/thomaslj/eegs-50epoch.err
#SBATCH --output=/scratch/users/thomaslj/eegs-50epoch.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH -p mignot
#SBATCH --gres gpu:1
#SBATCH -C GPU_MEM:16GB 

# Load the environment ,gpu,owners
cd $HOME/stroke-thomas
source ./src/prepare_env.sh

# python3 -m data.h5-generator 
python3 -m main -c config/param_configs_eegs.json