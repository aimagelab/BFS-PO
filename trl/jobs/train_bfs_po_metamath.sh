#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Experiment settings.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-FiorenzoParascandolo/MetaMathQA-60k}"
RUN_NAME="${RUN_NAME:-bfs_po_metamath}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/bfs_po_metamath}"
MAX_STEPS="${MAX_STEPS:-100000}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# Distributed resources
TOTAL_GPUS="${TOTAL_GPUS:-16}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
MACHINE_RANK="${MACHINE_RANK:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Select the launcher. TOTAL_GPUS must be divisible by GPUS_PER_NODE.
GPUS_PER_NODE=$((GPUS_PER_NODE < TOTAL_GPUS ? GPUS_PER_NODE : TOTAL_GPUS))
NUM_MACHINES=$((TOTAL_GPUS / GPUS_PER_NODE))
if ((NUM_MACHINES > 1)); then
    : "${MASTER_ADDR:?Set MASTER_ADDR}" "${MACHINE_RANK:?Set MACHINE_RANK}"
fi
LAUNCH_CMD=(python)
if ((TOTAL_GPUS > 1)); then
    LAUNCH_CMD=(accelerate launch
        --multi_gpu
        --num_processes "$TOTAL_GPUS"
        --num_machines "$NUM_MACHINES"
        --machine_rank "${MACHINE_RANK:-0}"
        --main_process_ip "${MASTER_ADDR:-127.0.0.1}"
        --main_process_port "$MASTER_PORT"
        --mixed_precision bf16
        --dynamo_backend no
    )
else
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
    set -- --resume_from_checkpoint "$RESUME_FROM_CHECKPOINT" "$@"
fi

# Launch training
exec "${LAUNCH_CMD[@]}" main_grpo.py \
    --grpo_config_path trl/config/GRPOConfig.yaml \
    --grpo_trainer_config_path trl/config/GRPOTrainer.yaml \
    "model=$MODEL_PATH" \
    "output_dir=$OUTPUT_DIR" \
    "run_name=$RUN_NAME" \
    'model_init_kwargs={dtype: bfloat16}' \
    "train_dataset_name=$TRAIN_DATASET_NAME" \
    'eval_dataset_name={math500: math-500}' \
    bfs_po_k=3 \
    branch_advantage=true \
    generation_batch_size=1024 \
    global_train_batch_size=512 \
    dynamic_sampling=true \
    per_device_train_batch_size=2 \
    per_device_eval_batch_size=32 \
    max_completion_length=4096 \
    max_completion_length_eval=4096 \
    eval_temperature=0.7 \
    top_p_eval=0.9 \
    epsilon_high=0.28 \
    vllm_gpu_memory_utilization=0.7 \
    vllm_max_model_length=7168 \
    lr_scheduler_type=constant_with_warmup \
    warmup_steps=160 \
    ddp_find_unused_parameters=false \
    eval_on_start=true \
    eval_strategy=steps \
    eval_steps=128 \
    save_steps=128 \
    save_total_limit=3 \
    load_best_model_at_end=true \
    metric_for_best_model=math500_aes \
    logging_steps=4 \
    "max_steps=$MAX_STEPS" \
    "$@"
