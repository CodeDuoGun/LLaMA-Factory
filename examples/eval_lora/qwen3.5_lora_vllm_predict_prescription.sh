#!/bin/bash

set -e
set -x

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-9B}
ADAPTER_PATH=${ADAPTER_PATH:-saves/qwen3.5-9b/lora/sft/prescription}
DATASET=${DATASET:-wuweiping_prescription}
DATASET_DIR=${DATASET_DIR:-data}
OUTPUT_DIR=${OUTPUT_DIR:-saves/qwen3.5-9b/lora/vllm_predict/prescription}
MAX_SAMPLES=${MAX_SAMPLES:-}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-}

mkdir -p "${OUTPUT_DIR}"

VLLM_CONFIG="{\"gpu_memory_utilization\": ${GPU_MEMORY_UTILIZATION}}"
if [[ -n "${TENSOR_PARALLEL_SIZE}" ]]; then
    VLLM_CONFIG="{\"gpu_memory_utilization\": ${GPU_MEMORY_UTILIZATION}, \"tensor_parallel_size\": ${TENSOR_PARALLEL_SIZE}}"
fi

ARGS=(
    --model_name_or_path "${MODEL_PATH}"
    --adapter_name_or_path "${ADAPTER_PATH}"
    --dataset "${DATASET}"
    --dataset_dir "${DATASET_DIR}"
    --template qwen3_5
    --cutoff_len 8192
    --save_name "${OUTPUT_DIR}/generated_predictions.jsonl"
    --matrix_save_name "${OUTPUT_DIR}/predict_results.json"
    --temperature 0.1
    --top_p 0.8
    --top_k 20
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --repetition_penalty 1.05
    --batch_size "${BATCH_SIZE}"
    --vllm_config "${VLLM_CONFIG}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
    ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

python scripts/vllm_infer.py "${ARGS[@]}"
