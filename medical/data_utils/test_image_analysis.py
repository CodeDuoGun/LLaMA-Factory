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

"""Run inspection and tongue/face image analysis independently for one record."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单病历图片解析诊断脚本；不运行病史或诊断清洗")
    parser.add_argument("--input", type=Path, required=True, help="病历 JSON 数组或 JSONL")
    parser.add_argument("--order-sn", required=True, help="要测试的病历 order_sn")
    parser.add_argument("--stage", choices=("inspection", "tongue_face", "both"), default="both")
    parser.add_argument("--classify", action="store_true", help="先按模型分类，再将分类结果送入对应解析协程")
    parser.add_argument("--model", default=os.getenv("MEDICAL_AGENT_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--vlm-model", default=os.getenv("MEDICAL_AGENT_VLM_MODEL", ""))
    parser.add_argument("--base-url", default=os.getenv("MEDICAL_AGENT_VLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    parser.add_argument("--api-key", default=os.getenv("MEDICAL_AGENT_VLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--timeout", type=float, default=60, help="每个解析协程最多等待的秒数")
    parser.add_argument("--max-retries", type=int, default=0)
    return parser.parse_args()


def load_record(path: Path, order_sn: str) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            records = (json.loads(line) for line in stream if line.strip())
            return next((record for record in records if record.get("order_sn") == order_sn), None) or _missing(order_sn)
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else next(
        (data[key] for key in ("data", "records", "items") if isinstance(data.get(key), list)), []
    )
    return next((record for record in records if record.get("order_sn") == order_sn), None) or _missing(order_sn)


def _missing(order_sn: str) -> dict[str, Any]:
    raise ValueError(f"未找到 order_sn={order_sn}")


async def run(args: argparse.Namespace) -> int:
    from medical.data_utils.online_record_processor import (
        TONGUE_FACE_IMAGE_CATEGORIES,
        MedicalRecordAgents,
        _as_images,
        build_model,
    )

    if not args.api_key:
        raise ValueError("缺少 VLM API Key；请设置 MEDICAL_AGENT_VLM_API_KEY 或 DASHSCOPE_API_KEY")
    record = load_record(args.input, args.order_sn)
    vlm = build_model(args.vlm_model or args.model, args, base_url=args.base_url, api_key=args.api_key)
    agents = MedicalRecordAgents(vlm, vlm, args.input.parent, stage_timeout=args.timeout)
    grouped = None
    if args.classify or args.stage in ("tongue_face", "both"):
        grouped = await asyncio.wait_for(agents.classify_images(record), timeout=args.timeout)
    if args.classify:
        inspection = grouped["检验检查报告类"]
    else:
        inspection = list(dict.fromkeys(
            url for field in ("inspection_report_img", "admin_report_img")
            for url in _as_images(record.get(field))
        ))
    tongue_face = {category: grouped[category] for category in TONGUE_FACE_IMAGE_CATEGORIES} if grouped else {}
    tongue_face_count = sum(map(len, tongue_face.values()))
    print(f"record={args.order_sn} inspection_images={len(inspection)} tongue_face_images={tongue_face_count}", flush=True)

    async def check(name: str, images: list[str] | dict[str, list[str]], output_field: str, method: Any) -> bool:
        image_count = sum(map(len, images.values())) if isinstance(images, dict) else len(images)
        if not image_count:
            print(f"{name}: 无图片，跳过", flush=True)
            return True
        working = dict(record)
        print(f"{name}: 开始，图片数={image_count}，超时={args.timeout:g}s", flush=True)
        try:
            await asyncio.wait_for(method(working, images), timeout=args.timeout)
        except Exception as exc:
            print(f"{name}: 失败 {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            return False
        print(f"{name}: 成功\n{json.dumps(working[output_field], ensure_ascii=False, indent=2)}", flush=True)
        return True

    checks = []
    if args.stage in ("inspection", "both"):
        checks.append(check("inspection", inspection, "ai_inspection_report_img", agents.analyze_inspection_images))
    if args.stage in ("tongue_face", "both"):
        checks.append(check("tongue_face", tongue_face, "ai_tongue_face_img", agents.analyze_tongue_face_images))
    results = await asyncio.gather(*checks)
    return 0 if all(results) else 1


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
