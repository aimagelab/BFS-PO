#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
cd -- "$REPO_DIR"
export REPO_DIR="$PWD"

export BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-1.7B}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_DIR/outputs/Qwen3-1.7B-BFS-PO}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/eval_outputs/final_test/BFS-PO/lora_checkpoint_template}"
export NUM_SEEDS="${NUM_SEEDS:-16}"
DATASETS=(math-500 gsm8k mmlu-stem amc2023)
DATASET_SPLITS='{"math-500": "test", "gsm8k": "test", "mmlu-stem": "test", "amc2023": "test"}'


echo "Evaluating base_model=$BASE_MODEL_PATH lora=$CHECKPOINT_PATH seeds=1..$NUM_SEEDS output=$OUTPUT_ROOT"

SEEDS=()
for ((seed = 1; seed <= NUM_SEEDS; seed++)); do
    SEEDS+=("$seed")
    # Extra-CoT: 4096-token limit; dont_consider_truncated_cot_in_length_count=true
    # excludes truncated CoTs with zero recorded length from the mean.
    # Example: 100, 200 tokens + a CoT truncated before </think> => mean 150 tokens.
    python main_eval.py \
        --model "$BASE_MODEL_PATH" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --processing_class_name "$CHECKPOINT_PATH" \
        --output_dir "$OUTPUT_ROOT/seed-$seed" \
        --save_metrics_path "$OUTPUT_ROOT/seed-$seed/metrics.json" \
        --datasets "${DATASETS[@]}" \
        --dataset_splits "$DATASET_SPLITS" \
        --seed "$seed" \
        --use_vllm \
        --vllm_model_impl vllm \
        --vllm_gpu_memory_utilization 0.80 \
        --per_device_eval_batch_size 256 \
        --generation_batch_size 256 \
        --num_generations 1 \
        --eval_sampling_strategy independent \
        --do_sample \
        --temperature 0.7 \
        --top_p 0.9 \
        --presence_penalty 0.0 \
        --max_completion_length 4096 \
        --dont_consider_truncated_cot_in_length_count true \
        --save_rendered_prompts \
        --save_generated_cot_examples_every_batch \
        "$@"
done

# Aggregate results after all requested seeds complete.
if ((NUM_SEEDS > 1)); then
    python "$REPO_DIR/trl/jobs/aggregate_seeds.py" \
        --input-dir "$OUTPUT_ROOT" \
        --seeds "${SEEDS[@]}"
else
    echo "Single-seed evaluation complete: metrics are in $OUTPUT_ROOT/seed-1/metrics.json."
    echo "Skipping aggregation: sample standard deviation requires at least two seeds."
fi
