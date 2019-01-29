#!/bin/bash

# echo "TensorFlow module loaded"

# echo "Activating conda environment"
#ml load py-tensorflow/1.6.0_py36
ml load py-tensorflow/1.6.0_py36
#ml load math cudnn
#ml load cuda/9.0.176
export PATH=/home/users/thomaslj/miniconda3/bin:$PATH
source activate deep-stroke