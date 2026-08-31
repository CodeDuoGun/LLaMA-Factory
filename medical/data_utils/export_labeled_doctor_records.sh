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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/medical/processed_data}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
CONDA_ENV="${CONDA_ENV:-llamafactory}"
AGENT_SCRIPT="${SCRIPT_DIR}/medical_record_agents.py"

usage() {
  echo "用法: $0 <doctor_id> <doctor_name> [output_dir]" >&2
  echo "示例: $0 43 朱子奇" >&2
  echo "      $0 43 朱子奇 /data/medical/processed_data" >&2
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

DOCTOR_ID="$1"
DOCTOR_NAME="$2"
if [[ -z "${DOCTOR_ID}" ]]; then
  echo "[ERROR] doctor_id 不能为空" >&2
  exit 2
fi
if [[ -z "${DOCTOR_NAME}" ]]; then
  echo "[ERROR] doctor_name 不能为空" >&2
  exit 2
fi
if [[ $# -eq 3 ]]; then
  OUTPUT_DIR="$3"
fi

if [[ ! -f "${AGENT_SCRIPT}" ]]; then
  echo "[ERROR] 导出脚本不存在: ${AGENT_SCRIPT}" >&2
  exit 1
fi

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
echo "[START] 导出 doctor_id=${DOCTOR_ID} 的全部 labeled 标注数据"
echo "[OUTPUT] ${OUTPUT_DIR}/doctor_${DOCTOR_ID}_${DOCTOR_NAME}/medical_records_labeled.jsonl"

cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${RUNNER[@]}" "${AGENT_SCRIPT}" \
  --data-source mysql \
  --export-mode \
  --doctor-id "${DOCTOR_ID}" \
  --export-status labeled \
  --export-doctor-name "${DOCTOR_NAME}" \
  --export-file-name medical_records_ai_stage.jsonl \
  --limit 0 \
  --output-dir "${OUTPUT_DIR}"

echo "[DONE] doctor_id=${DOCTOR_ID} 的 labeled 标注数据导出完成"
