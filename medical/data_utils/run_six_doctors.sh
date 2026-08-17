#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_DIR="${REPO_ROOT}/medical/data/202301_online"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/medical/processed_data}"

LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs/six_doctors}"
LIMIT_PER_DOCTOR="${LIMIT_PER_DOCTOR:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RECORD_TIMEOUT="${RECORD_TIMEOUT:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"

CONDA_BIN="${CONDA_BIN:-$(command -v conda)}"
CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-llama_factory}}"

AGENT_SCRIPT="${REPO_ROOT}/medical/data_utils/medical_record_agents.py"

PID_FILE="${LOG_DIR}/workers.pid"


mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
: > "${PID_FILE}"


if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
  echo "[ERROR] 未找到 conda，请通过 CONDA_BIN 指定 conda 可执行文件" >&2
  exit 1
fi


RUNNER=(
  "${CONDA_BIN}"
  run
  --no-capture-output
  -n
  "${CONDA_ENV}"
  python
)


if ! "${RUNNER[@]}" -c "import langchain, langchain_core" >/dev/null 2>&1; then
  echo "[ERROR] Conda 环境 ${CONDA_ENV} 不存在或未安装 langchain/langchain_core" >&2
  exit 1
fi


echo "[PYTHON] $("${RUNNER[@]}" -c "import sys; print(sys.executable)")"


# 支持多个stage
STAGES=(
  diagnosis_normalization
  diagnosis_completion
)


DOCTORS=(
  # "1314|吴卫平|吴卫平_AI医生分身混合问诊数据_1314_20260729180432.json"
  "97|巢国俊|巢国俊_AI医生分身混合问诊数据_97_20260729180345.json"
  "43|朱子奇|朱子奇_AI医生分身混合问诊数据_43_20260729175534.json"
  "567|李同新|李同新_AI医生分身混合问诊数据_567_20260729175826.json"
  "432|杨钦河|杨钦河_AI医生分身混合问诊数据_432_20260729180138.json"
  "577|郝建军|郝建军_AI医生分身混合问诊数据_577_20260729180249.json"
)


for doctor in "${DOCTORS[@]}"; do

  IFS="|" read -r doctor_id doctor_name input_file <<< "${doctor}"

  input_path="${INPUT_DIR}/${input_file}"
  log_path="${LOG_DIR}/doctor_${doctor_id}_${doctor_name}.log"


  if [[ ! -f "${input_path}" ]]; then
    echo "[ERROR] 输入文件不存在: ${input_path}" >&2
    exit 1
  fi


  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  nohup "${RUNNER[@]}" "${AGENT_SCRIPT}" \
    --data-source local \
    --save-to local \
    --input "${input_path}" \
    --output-dir "${OUTPUT_DIR}" \
    --doctor-id "${doctor_id}" \
    --limit "${LIMIT_PER_DOCTOR}" \
    --batch-size "${BATCH_SIZE}" \
    --record-timeout "${RECORD_TIMEOUT}" \
    --progress-every "${PROGRESS_EVERY}" \
    > "${log_path}" 2>&1 &


  pid=$!


  printf "%s\t%s\t%s\t%s\n" \
    "${pid}" \
    "${doctor_id}" \
    "${doctor_name}" \
    "${log_path}" \
    >> "${PID_FILE}"


  echo "[STARTED] pid=${pid} doctor=${doctor_id}/${doctor_name} log=${log_path}"

done


echo
echo "===================================="
echo "所有医生任务已启动"
echo "每个任务执行:"
echo "  diagnosis_normalization"
echo "  diagnosis_completion"
echo "===================================="

echo
echo "查看进程:"
echo "awk '{print \$1}' '${PID_FILE}' | xargs ps -p"

echo
echo "查看日志:"
echo "tail -f '${LOG_DIR}'/*.log"

echo
echo "查看进度:"
echo "tail -f '${LOG_DIR}'/*.log | grep --line-buffered '\\[PROGRESS\\]'"

echo
echo "停止任务:"
echo "awk '{print \$1}' '${PID_FILE}' | xargs kill"