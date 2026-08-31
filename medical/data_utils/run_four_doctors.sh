#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_DIR="${REPO_ROOT}/medical/data/202301_online"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/medical/processed_data}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs/four_doctors}"
PROGRESS_FILE="${PROGRESS_FILE:-${OUTPUT_DIR}/mysql_stage_update_progress.json}"

# 用法：
#   ./run_four_doctors.sh 1      每位医生从“未处理记录”中跑 1 条
#   ./run_four_doctors.sh 10     每位医生从“未处理记录”中跑 10 条
#   ./run_four_doctors.sh        每位医生全量跑剩余未处理记录
LIMIT_PER_DOCTOR="${1:-${LIMIT_PER_DOCTOR:-0}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RECORD_TIMEOUT="${RECORD_TIMEOUT:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-llama_factory}}"
AGENT_SCRIPT="${REPO_ROOT}/medical/data_utils/medical_record_agents.py"
PID_FILE="${LOG_DIR}/workers.pid"

mkdir -p "${LOG_DIR}" "$(dirname "${PROGRESS_FILE}")"

DOCTORS=(
  "97|巢国俊|巢国俊_AI医生分身混合问诊数据_97_20260729180345.json"
  "567|李同新|李同新_AI医生分身混合问诊数据_567_20260729175826.json"
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

: > "${PID_FILE}"

PIDS=()
DOCTOR_NAMES=()
DOCTOR_IDS=()

for doctor in "${DOCTORS[@]}"; do
  IFS='|' read -r DOCTOR_ID DOCTOR_NAME INPUT_FILE <<< "${doctor}"
  INPUT_PATH="${INPUT_DIR}/${INPUT_FILE}"
  LOG_FILE="${LOG_DIR}/doctor_${DOCTOR_ID}_${DOCTOR_NAME}.log"

  if [[ ! -f "${INPUT_PATH}" ]]; then
    echo "[ERROR] 原始数据不存在，跳过 ${DOCTOR_NAME}: ${INPUT_PATH}"
    continue
  fi

  CMD=(
    "${CONDA_BIN}" run
    --no-capture-output
    -n "${CONDA_ENV}"
    python "${AGENT_SCRIPT}"
    --data-source mysql
    --save-to mysql
    --input "${INPUT_PATH}"
    --doctor-id "${DOCTOR_ID}"
    --mysql-progress-file "${PROGRESS_FILE}"
    --stage-name
      image_classification
      inspection_vlm
      tongue_face_vlm
      clinical_extraction
    --batch-size "${BATCH_SIZE}"
    --record-timeout "${RECORD_TIMEOUT}"
    --progress-every "${PROGRESS_EVERY}"
  )

  if [[ "${LIMIT_PER_DOCTOR}" -gt 0 ]]; then
    CMD+=(--limit "${LIMIT_PER_DOCTOR}")
  fi

  echo "[START] ${DOCTOR_NAME} doctor_id=${DOCTOR_ID}"
  echo "[PROGRESS] ${PROGRESS_FILE}"
  echo "[LOG] ${LOG_FILE}"

  nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
  PID=$!

  PIDS+=("${PID}")
  DOCTOR_NAMES+=("${DOCTOR_NAME}")
  DOCTOR_IDS+=("${DOCTOR_ID}")
  echo "${PID}|${DOCTOR_ID}|${DOCTOR_NAME}|${LOG_FILE}" >> "${PID_FILE}"
  echo "[PID] ${PID}"
done

echo
echo "已启动 ${#PIDS[@]} 个后台进程"
echo "共享进度文件: ${PROGRESS_FILE}"
echo "PID 文件: ${PID_FILE}"
cat "${PID_FILE}"

echo
echo "等待所有任务完成..."
FAILED=0
for i in "${!PIDS[@]}"; do
  PID="${PIDS[$i]}"
  if wait "${PID}"; then
    echo "[OK] ${DOCTOR_NAMES[$i]} doctor_id=${DOCTOR_IDS[$i]} PID=${PID}"
  else
    EXIT_CODE=$?
    echo "[ERROR] ${DOCTOR_NAMES[$i]} doctor_id=${DOCTOR_IDS[$i]} PID=${PID} exit_code=${EXIT_CODE}"
    FAILED=$((FAILED + 1))
  fi
done

echo
echo "完成。失败进程数: ${FAILED}"
echo "进度文件: ${PROGRESS_FILE}"
exit "${FAILED}"
