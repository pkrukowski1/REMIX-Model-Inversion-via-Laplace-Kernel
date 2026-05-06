#!/bin/bash
#SBATCH --job-name=cl_laplace_kernel_generate_samples
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rtx4090_batch

export CUBLAS_WORKSPACE_CONFIG=:4096:8


python3 -u synthetic_samples_vit.py