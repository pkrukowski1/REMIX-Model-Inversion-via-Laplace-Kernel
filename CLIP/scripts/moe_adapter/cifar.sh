#!/bin/bash
#SBATCH --job-name=cl_laplace_kernel_moe_adapter_cifar_10x10
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=rtx4090

export CUBLAS_WORKSPACE_CONFIG=:4096:8

local_path='/shared/results/pkrukowski/LaplaceKernelInContinualLearning/CLIP/moe_adapter/'
mkdir -p "${local_path}"

mkdir -p /shared/results/pkrukowski/LaplaceKernelInContinualLearning/CLIP/moe_adapter/logs

cd $HOME/LaplaceKernelInContinualLearning/CLIP
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate laplace_kernel_in_cl

python3 -u main.py --config=./configs/moe_adapter/cf100.yaml \
	--seed 1993 \
	--local_path=$local_path