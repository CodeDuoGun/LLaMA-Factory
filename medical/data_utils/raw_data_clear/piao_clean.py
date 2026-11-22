# Copyright 2026 the LlamaFactory team.
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

"""Convert Piao's PDF records to numbered images and clean visits by date."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from medical.data_utils.raw_data_clear.zhangzongqi_clean import (  # noqa: E402
    MergedVisits,
    TextExtractionAgent,
    build_record,
    call_model,
    extract_image,
    load_prompts,
)
from medical.utils.log import logger  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "medical/data/朴老病历943"
DEFAULT_OUTPUT = PROJECT_ROOT / "medical/processed_data/doctor_朴老病历943"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PDF_SUFFIX = ".pdf"


def convert_pdfs_to_images(
    input_dir: Path,
    image_dir: Path,
    *,
    dpi: int,
    pdftoppm: str = "pdftoppm",
) -> dict[Path, list[Path]]:
    """Render each PDF page into a globally numbered PNG under the sibling img directory."""
    pdf_paths = sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == PDF_SUFFIX),
        key=lambda path: path.name.lower(),
    )
    if not pdf_paths:
        raise ValueError(f"目录中没有 PDF 文件: {input_dir}")
    if shutil.which(pdftoppm) is None:
        raise RuntimeError(f"找不到 PDF 转图工具 {pdftoppm!r}，请安装 Poppler 或通过 --pdftoppm 指定可执行文件")

    image_dir.mkdir(parents=True, exist_ok=True)
    images_by_pdf: dict[Path, list[Path]] = {}
    manifest: list[dict[str, Any]] = []
    sequence = 0

    for pdf_path in pdf_paths:
        with tempfile.TemporaryDirectory(prefix="piao_pdf_", dir=image_dir) as temporary_dir:
            prefix = Path(temporary_dir) / "page"
            result = subprocess.run(
                [pdftoppm, "-q", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                detail = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"PDF 转图失败: {pdf_path.name}: {detail}")

            rendered_pages = sorted(
                Path(temporary_dir).glob("page-*.png"),
                key=lambda path: int(path.stem.rsplit("-", 1)[1]),
            )
            if not rendered_pages:
                raise RuntimeError(f"PDF 未生成页面图片: {pdf_path}")

            pdf_images: list[Path] = []
            for page_number, page_path in enumerate(rendered_pages, start=1):
                sequence += 1
                image_path = image_dir / f"{sequence:06d}.png"
                page_path.replace(image_path)
                pdf_images.append(image_path)
                manifest.append(
                    {
                        "sequence": sequence,
                        "image": image_path.name,
                        "source_pdf": pdf_path.name,
                        "page": page_number,
                    }
                )
            images_by_pdf[pdf_path] = pdf_images
        logger.info(f"PDF 转图完成 pdf={pdf_path.name} pages={len(images_by_pdf[pdf_path])}")

    manifest_path = image_dir / "manifest.json"
    temporary_manifest = image_dir / ".manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    logger.info(f"PDF 转图全部完成 pdfs={len(pdf_paths)} images={sequence} manifest={manifest_path}")
    return images_by_pdf


def make_record(item: dict[str, Any], source_pdf: Path, visit_order: int, doctor_name: str) -> dict[str, Any]:
    """Map one structured visit to the existing records schema with a stable source-based order number."""
    record = build_record(item, "", visit_order)
    source_key = hashlib.sha256(source_pdf.name.encode("utf-8")).hexdigest()[:12]
    order_sn = f"piao_{source_key}_{visit_order:03d}"
    record["order_sn"] = order_sn
    record["doctor_name"] = doctor_name
    for prescription_index, prescription in enumerate(record["ps"], start=1):
        prescription["order_sn"] = order_sn
        prescription["prescription_order_id"] = f"{order_sn}_p{prescription_index:02d}"
    return record


def parse_args() -> argparse.Namespace:
    from dotenv import dotenv_values

    model_from_file = dotenv_values(PROJECT_ROOT / ".env").get("MEDICAL_AGENT_MODEL")
    default_model = model_from_file or os.getenv("MEDICAL_AGENT_MODEL", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT, help="PDF 所在目录")
    parser.add_argument("--image-dir", type=Path, help="图片目录；默认在 PDF 目录下创建 img/")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doctor-name", default="朴老")
    parser.add_argument("--vision-model", default=default_model)
    parser.add_argument("--text-model", default=default_model)
    parser.add_argument("--base-url", default=os.getenv("DASHSCOPE_BASE_URL", BASE_URL))
    parser.add_argument("--pdftoppm", default="pdftoppm", help="Poppler 的 pdftoppm 可执行文件")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    from medical.data_utils.structured_agent import AgentModelConfig, StructuredAgent

    if args.concurrency < 1 or args.retries < 0 or args.timeout <= 0 or args.dpi < 1:
        raise ValueError("concurrency、timeout 和 dpi 必须为正数，retries 不能为负数")
    if not args.vision_model or not args.text_model:
        raise ValueError("请在 .env 中设置 MEDICAL_AGENT_MODEL，或通过 --vision-model 和 --text-model 指定模型")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY")

    image_dir = args.image_dir or args.input_dir / "img"
    images_by_pdf = convert_pdfs_to_images(args.input_dir, image_dir, dpi=args.dpi, pdftoppm=args.pdftoppm)
    image_prompt, merge_prompt = load_prompts()
    image_prompt = image_prompt.replace("张宗岐", args.doctor_name)
    merge_prompt = merge_prompt.replace("张宗岐", args.doctor_name)

    def make_config(model: str, name: str, *, streaming: bool) -> AgentModelConfig:
        config = AgentModelConfig(
            model=model,
            api_key=api_key,
            base_url=args.base_url,
            temperature=0,
            thinking=False,
            timeout=args.timeout,
            max_retries=0,
            client_kwargs={"streaming": streaming},
        )
        logger.info(
            f"模型参数 agent={name} model={config.model} base_url={config.base_url} "
            f"temperature={config.temperature} thinking={config.thinking} streaming={streaming} "
            f"timeout={config.timeout} max_retries={config.max_retries} validation_retries=1"
        )
        return config

    image_config = make_config(args.vision_model, "piao_image_extractor", streaming=False)
    merge_config = make_config(args.text_model, "piao_visit_merger", streaming=True)
    image_agent = TextExtractionAgent(image_config.build(), image_prompt)
    merge_agent = StructuredAgent.from_config(
        merge_config,
        MergedVisits,
        system_prompt=merge_prompt,
        name="piao_visit_merger",
        output_mode="json_object",
        validation_retries=1,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    all_images = [image for pdf_images in images_by_pdf.values() for image in pdf_images]
    logger.info(
        f"开始朴老病历单图提取 pdfs={len(images_by_pdf)} images={len(all_images)} "
        f"concurrency={args.concurrency} vision_model={args.vision_model} text_model={args.text_model}"
    )

    async def limited_extract(path: Path) -> dict[str, Any]:
        async with semaphore:
            return await extract_image(image_agent, path, args.retries)

    extraction_results = await asyncio.gather(
        *(limited_extract(image_path) for image_path in all_images),
        return_exceptions=True,
    )
    extraction_failures = [
        (image_path, result)
        for image_path, result in zip(all_images, extraction_results)
        if isinstance(result, Exception)
    ]
    if extraction_failures:
        for image_path, error in extraction_failures:
            logger.error(f"单图解析失败 image={image_path.name}: {type(error).__name__}: {error}")
        logger.error(f"单图解析失败 images={len(extraction_failures)}/{len(all_images)}，未生成最终 JSONL")
        return 1

    extracted_by_image = {result["image"]: result for result in extraction_results}
    for pdf_path, pdf_images in images_by_pdf.items():
        extracted = [extracted_by_image[image_path.name] for image_path in pdf_images]
        raw_path = args.output_dir / f"{pdf_path.stem}_images.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    logger.info("单图提取全部完成，开始按 PDF 分组生成时间窗口病史及处方")

    async def merge_pdf(pdf_path: Path, pdf_images: list[Path]) -> list[dict[str, Any]]:
        extracted = [extracted_by_image[image_path.name] for image_path in pdf_images]
        async with semaphore:
            merged = await call_model(
                merge_agent,
                json.dumps(
                    {"source_pdf": pdf_path.name, "images": extracted},
                    ensure_ascii=False,
                ),
                args.retries,
            )
        if not isinstance(merged, dict) or not isinstance(merged.get("items"), list):
            raise ValueError(f"合并结果必须为包含 items 数组的 JSON 对象: {pdf_path.name}")
        if any(not isinstance(item, dict) for item in merged["items"]):
            raise ValueError(f"items 中存在非对象: {pdf_path.name}")
        records = [
            make_record(item, pdf_path, visit_order, args.doctor_name)
            for visit_order, item in enumerate(merged["items"], start=1)
        ]
        logger.info(f"PDF 病史结构化完成 pdf={pdf_path.name} images={len(pdf_images)} visits={len(records)}")
        return records

    pdf_paths = list(images_by_pdf)
    merge_results = await asyncio.gather(
        *(merge_pdf(pdf_path, images_by_pdf[pdf_path]) for pdf_path in pdf_paths),
        return_exceptions=True,
    )
    merge_failures = [
        (pdf_path, result) for pdf_path, result in zip(pdf_paths, merge_results) if isinstance(result, Exception)
    ]
    if merge_failures:
        for pdf_path, error in merge_failures:
            logger.error(f"病史合并失败 pdf={pdf_path.name}: {type(error).__name__}: {error}")
        logger.error(f"病史合并失败 pdfs={len(merge_failures)}/{len(pdf_paths)}，未覆盖最终 JSONL")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "medical_records_ai.jsonl"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output_dir, prefix=".medical_records_", suffix=".jsonl", delete=False
    ) as stream:
        temporary = Path(stream.name)
        for pdf_records in merge_results:
            for record in pdf_records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(target)
    record_count = sum(len(records) for records in merge_results)
    logger.info(f"病历写入完成 path={target} pdfs={len(pdf_paths)} records={record_count}")
    return 0


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
