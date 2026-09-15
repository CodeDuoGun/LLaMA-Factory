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
EXPORT_SCRIPT="${SCRIPT_DIR}/export_labeled_doctor_records.sh"

# 默认写入标注数据目录；也可以传入一个输出目录或设置 OUTPUT_DIR。
OUTPUT_DIR="${OUTPUT_DIR:-/Users/tangxueduo/Projects/tcm-disease-gateway/data/labeled_records}"
DELETE_OLD="${DELETE_OLD:-1}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
CONDA_ENV="${CONDA_ENV:-llamafactory}"

DOCTORS=(
  "567|李同新"
  "1314|吴卫平"
  "577|郝建军"
  "97|巢国俊"
  "43|朱子奇"
  "432|杨钦河"
)

usage() {
  echo "用法: $0 [output_dir]" >&2
  echo "默认输出目录: ${OUTPUT_DIR}" >&2
  echo "默认会删除每位医生的旧 medical_records_labeled.jsonl；设置 DELETE_OLD=0 可保留旧文件" >&2
}

if [[ $# -eq 1 && ("$1" == "-h" || "$1" == "--help") ]]; then
  usage
  exit 0
fi
if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi
if [[ $# -eq 1 ]]; then
  OUTPUT_DIR="$1"
fi
if [[ -z "${OUTPUT_DIR}" || "${OUTPUT_DIR}" == "/" ]]; then
  printf '[ERROR] 输出目录不能为空或根目录: "%s"\n' "${OUTPUT_DIR}" >&2
  exit 2
fi
case "${DELETE_OLD}" in
  0|1) ;;
  *)
    printf '[ERROR] DELETE_OLD 只能是 0 或 1，当前值: "%s"\n' "${DELETE_OLD}" >&2
    exit 2
    ;;
esac
if [[ ! -f "${EXPORT_SCRIPT}" ]]; then
  echo "[ERROR] 单医生导出脚本不存在: ${EXPORT_SCRIPT}" >&2
  exit 1
fi
if [[ -z "${CONDA_BIN}" || ! -x "${CONDA_BIN}" ]]; then
  echo "[ERROR] 未找到 conda，请通过 CONDA_BIN 指定 conda 可执行文件" >&2
  exit 1
fi
if ! "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python -c "import langchain, langchain_core" >/dev/null 2>&1; then
  echo "[ERROR] Conda 环境 ${CONDA_ENV} 不存在或未安装 langchain/langchain_core" >&2
  exit 1
fi
export CONDA_BIN CONDA_ENV

echo "[START] 导出 6 位医生的 labeled 病例标注数据"
echo "[OUTPUT] ${OUTPUT_DIR}"
echo "[DELETE_OLD] ${DELETE_OLD}"

for doctor in "${DOCTORS[@]}"; do
  IFS='|' read -r doctor_id doctor_name <<< "${doctor}"
  doctor_dir="${OUTPUT_DIR}/doctor_${doctor_id}_${doctor_name}"
  output_file="${doctor_dir}/medical_records_labeled.jsonl"

  if [[ "${DELETE_OLD}" == "1" ]]; then
    if [[ -d "${output_file}" ]]; then
      echo "[ERROR] 目标路径是目录，拒绝删除: ${output_file}" >&2
      exit 1
    fi
    if [[ -e "${output_file}" ]]; then
      rm -f -- "${output_file}"
      echo "[DELETE] ${output_file}"
    fi
  fi

  echo "[START] doctor_id=${doctor_id} doctor_name=${doctor_name}"
  bash "${EXPORT_SCRIPT}" "${doctor_id}" "${doctor_name}" "${OUTPUT_DIR}"
  echo "[DONE] doctor_id=${doctor_id} doctor_name=${doctor_name}"
done

echo "[DONE] 6 位医生的 labeled 病例标注数据导出完成"
