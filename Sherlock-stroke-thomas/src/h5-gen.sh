#!/bin/bash
#SBATCH --job-name=h5generatorthomas
#SBATCH --nodes=1
#SBATCH --error=/scratch/users/thomaslj/h5generatorthomas.err
#SBATCH --output=/scratch/users/thomaslj/h5generatorthomas.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -p mignot,normal,owners


# Load the environment ,gpu,owners
cd $HOME/stroke-thomas
source src/prepare_env.sh

python3 -m data.h5-generator 