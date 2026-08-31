#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_DIR="${REPO_ROOT}/medical/data/202301_online"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/medical/processed_data}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs/four_doctors}"
PROGRESS_FILE="${PROGRESS_FILE:-${OUTPUT_DIR}/mysql_stage_update_progress.json}"

# 用法：
# ./run_four_doctors.sh 1
# ./run_four_doctors.sh 10
# ./run_four_doctors.sh       # 全量剩余数据
LIMIT_PER_DOCTOR="${1:-${LIMIT_PER_DOCTOR:-0}}"

BATCH_SIZE="${BATCH_SIZE:-1}"
RECORD_TIMEOUT="${RECORD_TIMEOUT:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-llama_factory}}"

AGENT_SCRIPT="${REPO_ROOT}/medical/data_utils/medical_record_agents.py"

mkdir -p "${LOG_DIR}" "$(dirname "${PROGRESS_FILE}")"

DOCTORS=(
  # "97|巢国俊|巢国俊_AI医生分身混合问诊数据_97_20260729180345.json"
  # "567|李同新|李同新_AI医生分身混合问诊数据_567_20260729175826.json"
  # "432|杨钦河|杨钦河_AI医生分身混合问诊数据_432_20260729180138.json"
  "577|郝建军|郝建军_AI医生分身混合问诊数据_577_20260729180249.json"
)

if [[ -z "${CONDA_BIN}" ]]; then
  echo "[ERROR] 未找到 conda，请设置 CONDA_BIN"
  exit 1
fi

if [[ ! -f "${AGENT_SCRIPT}" ]]; then
  echo "[ERROR] Agent 脚本不存在: ${AGENT_SCRIPT}"
  exit 1
fi

echo "=================================================="
echo "Conda 环境     : ${CONDA_ENV}"
echo "Limit/doctor   : ${LIMIT_PER_DOCTOR}"
echo "Batch size     : ${BATCH_SIZE}"
echo "Progress file  : ${PROGRESS_FILE}"
echo "=================================================="

FAILED=0

for doctor in "${DOCTORS[@]}"; do
  IFS='|' read -r DOCTOR_ID DOCTOR_NAME INPUT_FILE <<< "${doctor}"

  INPUT_PATH="${INPUT_DIR}/${INPUT_FILE}"
  LOG_FILE="${LOG_DIR}/doctor_${DOCTOR_ID}_${DOCTOR_NAME}.log"

  if [[ ! -f "${INPUT_PATH}" ]]; then
    echo "[ERROR] 原始数据不存在: ${INPUT_PATH}"
    FAILED=$((FAILED + 1))
    continue
  fi

  echo
  echo "=================================================="
  echo "[START] ${DOCTOR_NAME}"
  echo "doctor_id : ${DOCTOR_ID}"
  echo "input     : ${INPUT_PATH}"
  echo "progress  : ${PROGRESS_FILE}"
  echo "log       : ${LOG_FILE}"
  echo "=================================================="

  LIMIT_ARGS=()

  if [[ "${LIMIT_PER_DOCTOR}" -gt 0 ]]; then
    LIMIT_ARGS=(--limit "${LIMIT_PER_DOCTOR}")
  fi

  "${CONDA_BIN}" run \
    --no-capture-output \
    -n "${CONDA_ENV}" \
    python "${AGENT_SCRIPT}" \
    --data-source mysql \
    --save-to mysql \
    --input "${INPUT_PATH}" \
    --doctor-id "${DOCTOR_ID}" \
    --mysql-progress-file "${PROGRESS_FILE}" \
    --stage-name \
      image_classification \
      inspection_vlm \
      tongue_face_vlm \
      clinical_cleaning \
    --batch-size "${BATCH_SIZE}" \
    --record-timeout "${RECORD_TIMEOUT}" \
    --progress-every "${PROGRESS_EVERY}" \
    "${LIMIT_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

  EXIT_CODE=${PIPESTATUS[0]}

  if [[ "${EXIT_CODE}" -eq 0 ]]; then
    echo
    echo "[OK] ${DOCTOR_NAME} doctor_id=${DOCTOR_ID} 处理完成"
  else
    echo
    echo "[ERROR] ${DOCTOR_NAME} doctor_id=${DOCTOR_ID} 处理失败 exit_code=${EXIT_CODE}"
    FAILED=$((FAILED + 1))
  fi
done

echo
echo "=================================================="
echo "全部任务结束"
echo "失败医生数 : ${FAILED}"
echo "进度文件   : ${PROGRESS_FILE}"
echo "日志目录   : ${LOG_DIR}"
echo "=================================================="

exit "${FAILED}"