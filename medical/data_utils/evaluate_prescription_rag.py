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

"""Run patient-isolated prescription RAG evaluation from the command line."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from medical.analysis.prescription_rag_eval import (
    DEFAULT_EVAL_ROOT,
    DEFAULT_INDEX_NAME,
    DEFAULT_OUTPUT_ROOT,
    EvalDatasetRepository,
    PrescriptionRAGEvaluator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行辨病Agent、开方Agent和开方RAG评测")
    parser.add_argument("--doctor-id", required=True)
    parser.add_argument("--doctor-name", default="")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--index", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--limit", type=int, default=0, help="最多评测多少条，0表示全部test样本")
    parser.add_argument("--top-k", type=int, default=10, help="传给开方Agent的相似病例数，至少10")
    parser.add_argument("--output", type=Path, help="结果JSON；默认写入analysis/output/prescription_rag_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit 不能小于0")
    if args.top_k < 10:
        raise SystemExit("--top-k 不能小于10，评测需要计算Recall@10和NDCG@10")
    datasets = EvalDatasetRepository(args.eval_root)
    info = datasets.info(args.doctor_id, args.doctor_name)
    if not info["available"]:
        raise SystemExit(f"未找到评测数据集，期望路径: {info['expected_path']}")
    if info["split_leaks"]:
        raise SystemExit(f"同一患者出现在多个split中: {info['split_leaks']}")
    cases = [
        case
        for case in datasets.load(Path(info["path"]))
        if str(case.get("split") or "test") == "test"
    ]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("没有可评测的split=test样本")

    result = PrescriptionRAGEvaluator().run(
        cases,
        args.index,
        top_k=args.top_k,
        progress=lambda done, total: print(f"[PROGRESS] {done}/{total}", flush=True),
    )
    output = args.output
    if output is None:
        doctor_label = f"doctor_{args.doctor_id}_{args.doctor_name}".rstrip("_")
        output = DEFAULT_OUTPUT_ROOT / doctor_label / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"结果已写入: {output}")


if __name__ == "__main__":
    main()

