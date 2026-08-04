#!/usr/bin/env bash
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-${REPO_ROOT}/medical/processed_data}"
LOG_DIR="${LOG_DIR:-${PROCESSED_DATA_DIR}/logs/normalize_ai_diagnosis}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OVERWRITE="${OVERWRITE:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
NORMALIZER_SCRIPT="${SCRIPT_DIR}/normalize_ai_diagnosis_labels.py"

DOCTORS=(
  "1314|吴卫平"
  "97|巢国俊"
  "43|朱子奇"
  "567|李同新"
  "432|杨钦河"
  "577|郝建军"
)

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] 未找到 Python: ${PYTHON_BIN}，请通过 PYTHON_BIN 指定可执行文件" >&2
  exit 1
fi
if [[ ! -f "${NORMALIZER_SCRIPT}" ]]; then
  echo "[ERROR] 归一化脚本不存在: ${NORMALIZER_SCRIPT}" >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c "import openpyxl" >/dev/null 2>&1; then
  echo "[ERROR] ${PYTHON_BIN} 所在环境未安装 openpyxl" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

# 先校验全部输入和输出，避免只启动部分医生任务。
for doctor in "${DOCTORS[@]}"; do
  IFS="|" read -r doctor_id doctor_name <<< "${doctor}"
  doctor_dir="${PROCESSED_DATA_DIR}/doctor_${doctor_id}_${doctor_name}"
  input_path="${doctor_dir}/medical_records_ai.jsonl"
  output_path="${doctor_dir}/medical_records_ai_normalized.jsonl"
  audit_path="${output_path}.audit.json"

  if [[ ! -f "${input_path}" ]]; then
    echo "[ERROR] 输入文件不存在: ${input_path}" >&2
    exit 1
  fi
  if [[ "${CHECK_ONLY}" != "1" && "${OVERWRITE}" != "1" && ( -e "${output_path}" || -e "${audit_path}" ) ]]; then
    echo "[ERROR] 输出或审计文件已存在: ${output_path}" >&2
    echo "        如需重新生成，请使用 OVERWRITE=1" >&2
    exit 1
  fi
done

PIDS=()
DOCTOR_LABELS=()
LOG_PATHS=()

terminate_workers() {
  echo "[STOPPING] 正在停止归一化子进程…" >&2
  for pid in "${PIDS[@]}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap terminate_workers INT TERM

for doctor in "${DOCTORS[@]}"; do
  IFS="|" read -r doctor_id doctor_name <<< "${doctor}"
  doctor_dir="${PROCESSED_DATA_DIR}/doctor_${doctor_id}_${doctor_name}"
  input_path="${doctor_dir}/medical_records_ai.jsonl"
  output_path="${doctor_dir}/medical_records_ai_normalized.jsonl"
  audit_path="${output_path}.audit.json"
  log_path="${LOG_DIR}/doctor_${doctor_id}_${doctor_name}.log"
  command_args=("${PYTHON_BIN}" "${NORMALIZER_SCRIPT}" --input "${input_path}")

  if [[ "${CHECK_ONLY}" == "1" ]]; then
    command_args+=(--check-only)
  else
    command_args+=(--output "${output_path}" --audit-output "${audit_path}")
    if [[ "${OVERWRITE}" == "1" ]]; then
      command_args+=(--overwrite)
    fi
  fi

  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${command_args[@]}" > "${log_path}" 2>&1 &
  pid=$!
  PIDS+=("${pid}")
  DOCTOR_LABELS+=("${doctor_id}/${doctor_name}")
  LOG_PATHS+=("${log_path}")
  echo "[STARTED] pid=${pid} doctor=${doctor_id}/${doctor_name} log=${log_path}"
done

failed=0
for index in "${!PIDS[@]}"; do
  pid="${PIDS[$index]}"
  doctor_label="${DOCTOR_LABELS[$index]}"
  log_path="${LOG_PATHS[$index]}"
  if wait "${pid}"; then
    echo "[SUCCESS] doctor=${doctor_label} log=${log_path}"
  else
    status=$?
    echo "[FAILED] doctor=${doctor_label} exit=${status} log=${log_path}" >&2
    failed=$((failed + 1))
  fi
done
trap - INT TERM

if (( failed > 0 )); then
  echo "[ERROR] ${failed} 个医生任务失败，请查看 ${LOG_DIR} 下的日志" >&2
  exit 1
fi

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "[DONE] 6 位医生的诊断标签检查完成，未写入 JSONL"
else
  echo "[DONE] 6 位医生的诊断标签归一化完成"
fi
