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

"""Extract Zhang Zongqi records image by image, then merge visits by date."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from medical.utils.log import logger  # noqa: E402


if TYPE_CHECKING:
    from medical.data_utils.structured_agent import StructuredAgent


DEFAULT_INPUT = PROJECT_ROOT / "medical/data/二期南区/zhangzongqi"
DEFAULT_OUTPUT = PROJECT_ROOT / "medical/processed_data/doctor_二期南区_张宗岐"
PROMPT_PATH = PROJECT_ROOT / "medical/data_utils/prompts/zhangzongqi_recordimg_clean.txt"
MERGE_PROMPT_PATH = PROJECT_ROOT / "medical/data_utils/prompts/zhangzongqi_record_merge_clean.txt"
RECORD_RULES_PATH = PROJECT_ROOT / "medical/data_utils/prompts/record_clear.txt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
IMAGE_NAME = re.compile(r"^(\d+)(?:_mask)?$", re.IGNORECASE)
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
STRING_FIELDS = (
    "order_sn",
    "inquiry_method",
    "is_first",
    "created_at",
    "start_time",
    "stop_time",
    "patient_name",
    "patient_sex",
    "patient_age",
    "patient_idcard",
    "patient_mobile",
    "patient_appeal",
    "doc_ass_stu_appeal",
    "new_medical_history",
    "is_old_medical_history",
    "old_medical_history",
    "is_allergic_history",
    "allergic_history",
    "is_personal_history",
    "personal_history",
    "is_special",
    "special_history",
    "is_child_history",
    "birth_detail",
    "is_marriage_history",
    "is_family_history",
    "family_history",
    "admin_report_describe",
    "admin_face_describe",
    "diagnosis_illness",
    "diagnosis_sickness",
    "diagnosis_disease",
    "disposition",
    "see_doc_time",
    "doctor_name",
    "ai_patient_appeal",
    "ai_new_medical_history",
    "ai_old_medical_history",
    "ai_allergic_history",
    "ai_personal_history",
    "ai_family_history",
    "ai_special_history",
    "ai_marriage_history",
    "ai_birth_detail",
    "ai_diagnosis_illness",
    "ai_diagnosis_sickness",
    "ai_diagnosis_disease",
    "ai_diagnosis_illness_reason",
    "ai_diagnosis_disease_reason",
    "ai_diagnosis_sickness_reason",
    "ai_disease_stage",
    "ai_disease_course",
    "ai_treatment_principle",
)
LIST_FIELDS = (
    "inspection_report_img",
    "tongue_face_img",
    "admin_report_img",
    "admin_face_img",
    "ps",
    "im_msg",
    "ai_etiology",
    "ai_pathogenesis",
    "ai_disease_location",
    "ai_key_symptoms",
)


class DrugItem(BaseModel):
    """One legible prescribed drug."""

    drug_name: str = ""
    drug_num: str = ""
    unit_name: str = ""


class PrescriptionItem(BaseModel):
    """One prescription for a visit."""

    usage_type: str = ""
    drug_process_name: str = ""
    use_limit: str = ""
    doctor_advice: str = ""
    usage_desc: str = ""
    prescription_items: dict[str, list[DrugItem]] = Field(default_factory=lambda: {"drugList": []})


class VisitItem(BaseModel):
    """Cleaned fields for exactly one encounter."""

    patient_name: str = ""
    patient_sex: str = ""
    is_first: str = ""
    see_doc_time: str = ""
    patient_age: str = ""
    patient_appeal: str = ""
    new_medical_history: str = ""
    old_medical_history: str = ""
    allergic_history: str = ""
    personal_history: str = ""
    special_history: str = ""
    birth_detail: str = ""
    marriage_history: str = ""
    family_history: str = ""
    admin_face_describe: str = ""
    admin_report_describe: str = ""
    diagnosis_illness: str = ""
    diagnosis_disease: str = ""
    diagnosis_sickness: str = ""
    ps: list[PrescriptionItem] = Field(default_factory=list)


class MergedVisits(BaseModel):
    """Visits merged from all images of one patient."""

    items: list[VisitItem]


def load_prompts(image_path: Path = PROMPT_PATH, merge_path: Path = MERGE_PROMPT_PATH) -> tuple[str, str]:
    image_prompt = image_path.read_text(encoding="utf-8").strip()
    merge_base = merge_path.read_text(encoding="utf-8").strip()
    if not image_prompt or not merge_base:
        raise ValueError(f"单图或合并提示词为空: {image_path}, {merge_path}")
    rules = RECORD_RULES_PATH.read_text(encoding="utf-8")
    clinical_rules = rules.split("【主诉清洗规则】", 1)[1].split("【三个诊断清洗规则】", 1)[0]
    history_rules = rules.split("【过敏史清洗过则】", 1)[1].split("【结构化输出】", 1)[0]
    merge_prompt = (
        merge_base + "\n\n以下是 record_clear.txt 中主诉、现病史及病史字段的详细规则；"
        "其诊断字段映射和原输出格式不适用于本任务：\n【主诉清洗规则】"
        + clinical_rules
        + "\n【过敏史清洗过则】"
        + history_rules
    )
    return image_prompt, merge_prompt


def select_images(patient_dir: Path) -> list[Path]:
    """Select the masked image first for each numeric sequence."""
    selected: dict[int, Path] = {}
    for path in sorted(patient_dir.iterdir(), key=lambda item: item.name.lower()):
        match = IMAGE_NAME.fullmatch(path.stem)
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES or not match:
            continue
        number = int(match.group(1))
        current = selected.get(number)
        if current is None or (path.stem.lower().endswith("_mask") and not current.stem.lower().endswith("_mask")):
            selected[number] = path
    return [selected[number] for number in sorted(selected)]


def clinical_text(value: Any) -> str:
    """Keep clinical details readable in fields that expect strings."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "；".join(part for item in value if (part := clinical_text(item)))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return ""


def build_record(item: dict[str, Any], patient_name: str, visit_order: int) -> dict[str, Any]:
    """Map one merged visit onto the existing medical_records_ai.jsonl schema."""
    record = dict.fromkeys(STRING_FIELDS, "")
    record.update({field: [] for field in LIST_FIELDS})
    record.update(
        {"user_info_id": None, "patient_id": 0, "patient_height": None, "patient_weight": None, "doctor_id": 0}
    )
    patient_key = hashlib.sha256(patient_name.encode("utf-8")).hexdigest()[:12]
    order_sn = f"zhangzongqi_{patient_key}_{visit_order:03d}"
    visit_date = clinical_text(item.get("see_doc_time"))
    record.update(
        {
            "order_sn": order_sn,
            "inquiry_method": "门诊",
            "created_at": visit_date,
            "start_time": visit_date,
            "patient_name": patient_name,
            "doctor_name": "张宗岐",
        }
    )
    for field in VisitItem.model_fields:
        if field == "ps":
            continue
        if field == "marriage_history":
            record["is_marriage_history"] = clinical_text(item.get(field))
        elif field == "patient_name":
            record[field] = clinical_text(item.get(field)) or patient_name
        else:
            record[field] = clinical_text(item.get(field))
    record["doc_ass_stu_appeal"] = record["patient_appeal"]
    record["ps"] = [
        {"id": None, "prescription_order_id": f"{order_sn}_p{index:02d}", "order_sn": order_sn, **prescription}
        for index, prescription in enumerate(item.get("ps", []), 1)
    ]
    record["is_old_medical_history"] = "是" if record["old_medical_history"] else ""
    for field in ("allergic", "personal", "family"):
        record[f"is_{field}_history"] = "是" if record[f"{field}_history"] else ""
    record["is_special"] = "是" if record["special_history"] else ""
    return record


async def call_model(agent: StructuredAgent[Any], content: Any, retries: int) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    def chunk_text(chunk: Any) -> str:
        value = chunk.content
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(
                block if isinstance(block, str) else str(block.get("text", ""))
                for block in value
                if isinstance(block, str) or (isinstance(block, dict) and block.get("type") in {"text", "output_text"})
            )
        return ""

    for attempt in range(retries + 1):
        try:
            messages = [SystemMessage(content=agent._json_system_prompt()), HumanMessage(content=content)]
            parts = []
            async for chunk in agent.runnable.astream(messages):
                parts.append(chunk_text(chunk))
            raw_text = "".join(parts).strip()
            if not raw_text:
                raise ValueError("合并 agent 的流式响应为空")
            fence = chr(96) * 3
            if raw_text.startswith(fence) and raw_text.endswith(fence):
                raw_text = re.sub(r"^" + fence + r"(?:json)?\s*|\s*" + fence + r"$", "", raw_text, flags=re.I).strip()
            return agent.schema.model_validate(json.loads(raw_text)).model_dump()
        except Exception as exc:
            if attempt == retries:
                logger.error(f"模型调用失败 agent={agent.name} attempts={attempt + 1}: {type(exc).__name__}: {exc}")
                raise
            logger.warning(
                f"模型调用重试 agent={agent.name} attempt={attempt + 1}/{retries + 1}: {type(exc).__name__}: {exc}"
            )
            await asyncio.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


class TextExtractionAgent:
    """Read one image as plain clinical text."""

    def __init__(self, model: Any, prompt: str) -> None:
        self.model = model
        self.prompt = prompt
        self.name = "zhangzongqi_image_extractor"

    async def ainvoke(self, content: Any) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = await self.model.ainvoke([SystemMessage(content=self.prompt), HumanMessage(content=content)])
        if not isinstance(response.content, str) or not response.content.strip():
            raise ValueError("单图模型未返回纯文本")
        return response.content.strip()


async def extract_image(agent: TextExtractionAgent, path: Path, retries: int) -> dict[str, Any]:
    sequence = int(IMAGE_NAME.fullmatch(path.stem).group(1))
    logger.info(f"开始解析图片 patient={path.parent.name} sequence={sequence} image={path.name}")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    content = [
        {"type": "text", "text": "逐字提取本图有效医学病史和处方，返回纯文本。"},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
    ]
    for attempt in range(retries + 1):
        try:
            result = await agent.ainvoke(content)
            break
        except Exception as exc:
            if attempt == retries:
                logger.error(f"单图解析失败 sequence={sequence} image={path.name}: {type(exc).__name__}: {exc}")
                raise
            logger.warning(f"单图解析重试 sequence={sequence} image={path.name} attempt={attempt + 1}: {exc}")
            await asyncio.sleep(min(2**attempt, 8))
    logger.info(f"单图解析结果 patient={path.parent.name} sequence={sequence} image={path.name}\n{result}")
    return {"image": path.name, "sequence": sequence, "extraction": result}


async def process_patient(
    image_agent: TextExtractionAgent,
    merge_agent: StructuredAgent[MergedVisits],
    patient_dir: Path,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    retries: int,
) -> list[dict[str, Any]]:
    images = select_images(patient_dir)
    if not images:
        logger.warning(f"患者目录无可处理图片 patient={patient_dir.name}")
        return []
    logger.info(f"开始清洗患者 patient={patient_dir.name} images={len(images)}")

    async def limited_extract(path: Path) -> dict[str, Any]:
        async with semaphore:
            return await extract_image(image_agent, path, retries)

    extracted = await asyncio.gather(*(limited_extract(path) for path in images))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{patient_dir.name}_images.json").write_text(
        json.dumps(extracted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"单图解析全部完成 patient={patient_dir.name} images={len(extracted)}，开始合并就诊窗口")
    merged = await call_model(
        merge_agent,
        json.dumps({"patient_folder": patient_dir.name, "images": extracted}, ensure_ascii=False),
        retries,
    )
    if not isinstance(merged, dict) or not isinstance(merged.get("items"), list):
        raise ValueError(f"合并结果必须为包含 items 数组的 JSON 对象: {patient_dir.name}")
    if any(not isinstance(item, dict) for item in merged["items"]):
        raise ValueError(f"items 中存在非对象: {patient_dir.name}")
    records = [build_record(item, patient_dir.name, index) for index, item in enumerate(merged["items"], 1)]
    logger.info(f"患者清洗完成 patient={patient_dir.name} images={len(images)} visits={len(records)}")
    return records


def parse_args() -> argparse.Namespace:
    from dotenv import dotenv_values

    model_from_file = dotenv_values(PROJECT_ROOT / ".env").get("MEDICAL_AGENT_MODEL")
    default_model = model_from_file or os.getenv("MEDICAL_AGENT_MODEL", "")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--patient", help="只处理指定患者文件夹；默认处理所有患者")
    parser.add_argument("--vision-model", default=default_model)
    parser.add_argument("--text-model", default=default_model)
    parser.add_argument("--base-url", default=os.getenv("DASHSCOPE_BASE_URL", BASE_URL))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    from medical.data_utils.structured_agent import AgentModelConfig, StructuredAgent, model_json_schema

    if args.concurrency < 1 or args.retries < 0 or args.timeout <= 0:
        raise ValueError("concurrency 和 timeout 必须为正数，retries 不能为负数")
    if not args.vision_model or not args.text_model:
        raise ValueError("请在 .env 中设置 MEDICAL_AGENT_MODEL，或通过 --vision-model 和 --text-model 指定模型")
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("请设置 DASHSCOPE_API_KEY")
    image_prompt, merge_prompt = load_prompts()
    if args.patient:
        patient_dirs = [args.input_dir / args.patient]
    else:
        patient_dirs = sorted((path for path in args.input_dir.iterdir() if path.is_dir()), key=lambda path: path.name)
    if not patient_dirs or any(not path.is_dir() for path in patient_dirs):
        raise ValueError("未找到患者目录")
    logger.info(
        f"开始张宗岐病历清洗 patients={len(patient_dirs)} concurrency={args.concurrency} "
        f"vision_model={args.vision_model} text_model={args.text_model}"
    )

    def make_config(model: str, name: str, *, streaming: bool) -> AgentModelConfig:
        config = AgentModelConfig(
            model=model,
            api_key=key,
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

    image_config = make_config(args.vision_model, "zhangzongqi_image_extractor", streaming=False)
    merge_config = make_config(args.text_model, "zhangzongqi_visit_merger", streaming=True)
    image_agent = TextExtractionAgent(image_config.build(), image_prompt)
    logger.debug(f"合并结果 schema={model_json_schema(MergedVisits).get('title')}")
    merge_agent = StructuredAgent.from_config(
        merge_config,
        MergedVisits,
        system_prompt=merge_prompt,
        name="zhangzongqi_visit_merger",
        output_mode="json_object",
        validation_retries=1,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *(
            process_patient(
                image_agent,
                merge_agent,
                patient_dir,
                args.output_dir,
                semaphore,
                args.retries,
            )
            for patient_dir in patient_dirs
        ),
        return_exceptions=True,
    )
    failures = [(path.name, result) for path, result in zip(patient_dirs, results) if isinstance(result, Exception)]
    for name, error in failures:
        logger.error(f"患者清洗失败 patient={name}: {type(error).__name__}: {error}")
    if failures:
        logger.error(f"本次清洗失败 patients={len(failures)}/{len(patient_dirs)}，未覆盖最终 JSONL")
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "medical_records_ai.jsonl"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output_dir, prefix=".medical_records_", suffix=".jsonl", delete=False
    ) as stream:
        temporary = Path(stream.name)
        for patient_records in results:
            for record in patient_records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(target)
    logger.info(f"病历写入完成 path={target} records={sum(map(len, results))}")
    return 0


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
