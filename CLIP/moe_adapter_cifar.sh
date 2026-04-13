export CUBLAS_WORKSPACE_CONFIG=:4096:8

local_path='/shared/results/pkrukowski/LaplaceKernelInContinualLearning/CLIP/moe_adapter'
mkdir -p "${local_path}"

python3 -u main.py --config=./configs/moe_adapter/cf100.yaml \
	--seed 1993 \
	--local_path=$local_path
