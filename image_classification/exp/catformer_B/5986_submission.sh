#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=/home/u000011200011/data/CATv2/CATFormer-D4,5,12,4-C64,128,320,512-H2,4,8,16-K5,5,1,1-5-small-plus-1-1-16,1-4,1,1-0.2-/image_classification/exp/uniformer_small/%j_0_log.err
#SBATCH --gpus-per-node=4
#SBATCH --job-name=convnext
#SBATCH --mem=160GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --open-mode=append
#SBATCH --output=/home/u000011200011/data/CATv2/CATFormer-D4,5,12,4-C64,128,320,512-H2,4,8,16-K5,5,1,1-5-small-plus-1-1-16,1-4,1,1-0.2-/image_classification/exp/uniformer_small/%j_0_log.out
#SBATCH --partition=gpu-a800
#SBATCH --signal=USR2@120
#SBATCH --time=4320
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/u000011200011/data/CATv2/CATFormer-D4,5,12,4-C64,128,320,512-H2,4,8,16-K5,5,1,1-5-small-plus-1-1-16,1-4,1,1-0.2-/image_classification/exp/uniformer_small/%j_%t_log.out --error /home/u000011200011/data/CATv2/CATFormer-D4,5,12,4-C64,128,320,512-H2,4,8,16-K5,5,1,1-5-small-plus-1-1-16,1-4,1,1-0.2-/image_classification/exp/uniformer_small/%j_%t_log.err /home/u000011200011/.conda/envs/py3.9/bin/python -u -m submitit.core._submit /home/u000011200011/data/CATv2/CATFormer-D4,5,12,4-C64,128,320,512-H2,4,8,16-K5,5,1,1-5-small-plus-1-1-16,1-4,1,1-0.2-/image_classification/exp/uniformer_small
