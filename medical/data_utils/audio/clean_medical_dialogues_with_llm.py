#!/usr/bin/env python3
"""使用 OpenAI 兼容 LLM 清洗皮肤科初诊/复诊医生-患者对话数据。

支持 JSON 数组和 JSONL 输入，并自动识别以下对话字段：
  - messages / conversations: [{"role": "user|assistant", "content": "..."}]
  - dialogue: [{"speaker": "患者|医生", "content": "..."}]
  - cleared_data.dialogue: 同上

输出：
  - --output：保留原记录结构，仅替换清洗后的对话字段。
  - --problem-ids-output：非对话、非皮肤科或质量不合格而被删除的病历 ID 及原因。
  - --error-output：接口失败或模型输出不合法、因而未进入结果的数据。

示例：
  export OPENAI_API_KEY='...'
  python tools/clean_medical_dialogues_with_llm.py \
    --input raw.json --output cleaned.json \
    --problem-ids-output problem_ids.json \
    --model qwen-plus --base-url https://dashscope.aliyuncs.com/compatible-mode/v1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable
from dotenv import load_dotenv

load_dotenv()



SYSTEM_PROMPT = """你是皮肤科医患对话数据清洗专家。输入是一组按时间排列的医生-患者对话，清洗结果将用于监督微调（SFT），目标是让模型学习专业、自然、循序渐进的皮肤科初诊和复诊问诊风格，而不是学习编造诊断或处方。

角色约定：
- user 永远是患者（或代患者陈述的家属）。
- assistant 永远是医生。

总体原则：只做有原文依据的删减、角色纠正、脱敏和轻度口语整理。不得为了使对话完整而新增原文没有的症状、体征、皮损描述、检查结果、诊断、药物、剂量、疗效、病史或建议；无法可靠修复时宁可 keep=false，也不要猜测。

你必须同时执行以下规则：

一、任务范围与对话有效性
1. 只保留皮肤科初诊或复诊问诊。皮肤、毛发、甲、皮肤附属器以及与皮肤科直接相关的黏膜问题属于范围内。明显属于其他科室且没有皮肤科相关主诉的对话返回 keep=false，并标记 out_of_scope。
2. 输入必须是真实的双向医患对话。病历摘要、处方、检查报告、单人独白、科普文章、机械问答列表、仅有单一角色、无法形成上下文的碎片，均返回 keep=false，并标记 not_dialogue。
3. 至少保留患者和医生各一轮。允许对话从初诊或复诊的任意合理阶段开始，但上下文必须足以理解医生正在追问什么。代词指向完全缺失、答非所问且无法判断原意、关键轮次缺失导致无法恢复的对话返回 keep=false，并标记 context_break。

二、文本、重复与角色清洗
4. 删除医生话语内部的重复句子、重复短语、口头禅和无意义复述，但保留全部不同的医学信息，标记 doctor_repetition。
5. 删除患者话语内部的重复内容；将“那个那个那个”“就是就是”“然后然后”“嗯嗯嗯”等连续重复语气词压缩为零个或一个，标记 patient_repetition。口语可以轻度整理，但不能改变肯定/否定、时间、部位、程度和原意。
6. 清理明显的转写噪声、错误标点、说话人标签及不影响医学含义的口误，标记 asr_noise。对药名、疾病名、数值、频次、剂量、单位、否定词、时间和左右侧只有在上下文能够唯一确定时才允许纠正；不能唯一确定则不得猜测，严重影响语义时返回 keep=false。
7. 根据完整上下文判断每句话真正属于患者还是医生并纠正角色错位，标记 role_confusion。医生负责询问、确认、解释和提出建议；患者负责陈述病情、回答和向医生提问。禁止医患相邻地询问或回答同一个问题。若医生错误地抢说了紧随其后的患者问题，将该医生轮次改成简短自然的承接语，如“好的”或“明白了”，保留患者的真实问题。
8. 删除与本次就诊无关的闲聊、广告、营销、引流、系统提示和模板污染，标记 template_noise；但不要删除能体现自然问诊衔接的简短问候、确认和共情。

三、皮肤科医学一致性
9. 检查同一对话内年龄、性别、孕产情况、皮损部位、左右侧、起病时间、病程、症状变化、过敏史、既往史、用药名称、用法、疗效和不良反应是否前后一致。存在无法由上下文解释的关键医学矛盾时，不要替患者选择一个版本，返回 keep=false，并标记 medical_inconsistency。
10. 医生不得凭空声称看到了原文未提供的皮损、照片、皮肤镜、化验、病理或其他检查结果；不得把可能诊断说成已确诊；不得新增患者未陈述的信息。发现此类无依据内容，若删除相关内容后仍能形成完整问诊则删除并标记 unsupported_claim，否则返回 keep=false。
11. 删除明显危险、错误或不专业的医疗建议，并标记 unsafe_advice，例如无依据保证治愈、鼓励自行停用关键药物、忽视严重药物过敏或感染风险、用偏方替代必要诊疗。若危险建议是核心回复且删除后对话失去意义，返回 keep=false。出现呼吸困难、意识异常、面唇喉快速肿胀、广泛水疱或皮肤剥脱、高热伴迅速扩散皮疹等危险信号却被医生明显忽视时，也返回 keep=false。
12. 必须彻底脱敏。将姓名、手机号、身份证号、详细地址、单位/学校、微信号、邮箱、病案号、住院号、医保号和其他可识别身份的信息替换为对应占位符，如“[姓名]”“[手机号]”“[病历号]”，并标记 privacy。若隐私信息过多、无法可靠脱敏或脱敏后对话不可理解，返回 keep=false。

四、要保留和鼓励的皮肤科问诊风格
13. 初诊重点是围绕患者已给出的线索逐步追问，可包括：主要皮肤问题、首次出现时间、发生部位和分布、皮损形态及颜色、数量和范围、瘙痒/疼痛/灼热/渗出等感觉、持续或反复、扩散及变化、诱因和接触史、季节环境关系、既往处理与效果、伴随全身症状、既往皮肤病、过敏史、用药史、家族史，以及在确有必要时询问妊娠哺乳和生活习惯。不要为了覆盖槽位而凭空补问或补答，也不要求每组对话覆盖所有内容。
14. 复诊重点是基于既往诊疗信息追问：与上次相比的皮损和症状变化、当前治疗的执行情况、药物或外用制剂的名称/用法、疗效、不良反应、停药或漏用原因、新发皮损或新症状、复查结果和患者当前诉求。不得把复诊改写成重复初诊，也不得编造上次诊断、处方或疗效。
15. 医生应围绕患者上一轮信息进行针对性追问，语气专业、尊重、克制，可以有简短确认和共情。高质量样本中，医生单轮应以一个核心问题为主，通常不超过二至三个紧密相关的子问题；删除已经得到答案的机械重复提问。不要删除患者后文实际回答了的不同问题，也不要擅自拆分或合并轮次。若原文是一次罗列大量问题的机械问卷且无法通过删除重复项安全修复，返回 keep=false，并标记 poor_doctor_style。
16. 对于目标为“问诊风格学习”的数据，优先保留信息采集、澄清、复诊疗效评估和风险筛查。医生过早下确定诊断、在信息不足时直接开药、夸大结论、训斥患者、过度口语化或答非所问，标记 poor_doctor_style；能够仅删除问题内容并保留连贯问诊时可修复，否则返回 keep=false。

五、输出结构
17. 保持原来的轮次顺序和 user/assistant 角色体系，不要合并或拆分轮次。允许删除纯噪声轮次，但不要编造轮次。每个保留轮次的 content 必须非空。
18. issue_types 只填写实际发现的问题。可用标签仅限：doctor_repetition、patient_repetition、role_confusion、not_dialogue、out_of_scope、context_break、asr_noise、template_noise、medical_inconsistency、unsupported_claim、unsafe_advice、privacy、poor_doctor_style。

典型纠错：
输入中医生说“运动可以做吗？”，紧接着患者说“增肌的运动可以做吗？”，说明前一句抢说了患者问题；应把医生轮次改成“好的”，保留患者的完整问题。

只返回一个 JSON 对象，不要返回 Markdown：
{
  "keep": true或false,
  "cleaned_dialogue": [{"role": "user或assistant", "content": "清洗后的文本"}],
  "issue_types": ["从允许的标签中选择实际存在的问题"],
  "reason": "一句话说明处理依据"
}
keep=false 时 cleaned_dialogue 必须为空数组。issue_types 只填写实际发现的问题；无问题时为空数组。"""

ROLE_ALIASES = {
    "user": "user",
    "patient": "user",
    "患者": "user",
    "病人": "user",
    "家属": "user",
    "assistant": "assistant",
    "doctor": "assistant",
    "医生": "assistant",
    "医师": "assistant",
    "大夫": "assistant",
}
ALLOWED_ISSUES = {
    "doctor_repetition",
    "patient_repetition",
    "role_confusion",
    "not_dialogue",
    "out_of_scope",
    "context_break",
    "asr_noise",
    "template_noise",
    "medical_inconsistency",
    "unsupported_claim",
    "unsafe_advice",
    "privacy",
    "poor_doctor_style",
}
ID_FIELDS = ("record_id", "medical_record_id", "case_id", "id", "病历id", "病历ID")
_thread_local = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用 LLM 清洗医患对话数据。")
    parser.add_argument("--input", required=True, help="输入 .json 或 .jsonl 文件。")
    parser.add_argument("--output", required=True, help="清洗后文件；扩展名决定 JSON/JSONL 格式。")
    parser.add_argument(
        "--problem-ids-output",
        required=True,
        help="被删除的问题数据病历 ID 及原因 JSON 文件。",
    )
    parser.add_argument(
        "--error-output",
        help="错误记录 JSONL；默认写到 output 同目录的 <stem>_errors.jsonl。",
    )
    parser.add_argument("--model", default=os.getenv("QWEN_TEXT_MODEL"), help="LLM 模型名。")
    parser.add_argument("--base-url", default=os.getenv("REPORT_VLM_BASE_URL"), help="OpenAI 兼容 base URL。")
    parser.add_argument("--api-key", default=os.getenv("REPORT_VLM_API_KEY"), help="默认读取 OPENAI_API_KEY。")
    parser.add_argument("--id-field", help="指定病历 ID 字段；默认自动识别。")
    parser.add_argument("--workers", type=int, default=4, help="并发请求数，默认 4。")
    parser.add_argument("--retries", type=int, default=3, help="每条记录最大请求次数，默认 3。")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次请求超时秒数。")
    parser.add_argument("--max-turn-chars", type=int, default=4000, help="单轮最大输入字符数。")
    parser.add_argument("--limit", type=int, help="仅处理前 N 条，便于试跑。")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    if not args.model:
        parser.error("必须通过 --model 或 OPENAI_MODEL 指定模型")
    if not args.api_key:
        parser.error("必须通过 --api-key 或 OPENAI_API_KEY 提供密钥")
    if args.workers < 1 or args.retries < 1 or args.max_turn_chars < 1:
        parser.error("--workers、--retries 和 --max-turn-chars 必须大于 0")
    return args


def load_records(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_no} 不是合法 JSON") from exc
        return records
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError("JSON 输入的顶层必须是数组")
    return value


def write_records(path: Path, records: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(records)
    with path.open("w", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        else:
            json.dump(values, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def record_id(record: Any, index: int, id_field: str | None) -> str:
    if isinstance(record, dict):
        fields = (id_field,) if id_field else ID_FIELDS
        for field in fields:
            if field and record.get(field) not in (None, ""):
                return str(record[field])
    return f"__index_{index}"


def normalize_role(value: Any) -> str | None:
    return ROLE_ALIASES.get(str(value).strip().lower())


def find_dialogue(record: Any, max_turn_chars: int) -> tuple[list[dict[str, str]], str]:
    """返回规范化对话及原字段路径标记。"""
    if not isinstance(record, dict):
        return [], "missing"
    candidates = (
        ("messages", record.get("messages")),
        ("conversations", record.get("conversations")),
        ("dialogue", record.get("dialogue")),
        (
            "cleared_data.dialogue",
            record.get("cleared_data", {}).get("dialogue")
            if isinstance(record.get("cleared_data"), dict)
            else None,
        ),
    )
    for field_path, raw_turns in candidates:
        if not isinstance(raw_turns, list) or not raw_turns:
            continue
        turns: list[dict[str, str]] = []
        for turn in raw_turns:
            if not isinstance(turn, dict):
                continue
            role = normalize_role(turn.get("role", turn.get("speaker")))
            content = turn.get("content", turn.get("text", ""))
            content = str(content).strip() if content is not None else ""
            if role and content:
                turns.append({"role": role, "content": content[:max_turn_chars]})
        return turns, field_path
    return [], "missing"


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("模型返回中没有 JSON 对象")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("模型返回的 JSON 不是对象")
    return value


def validate_result(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value.get("keep"), bool):
        raise ValueError("keep 必须是布尔值")
    reason = str(value.get("reason", "")).strip()
    issues = value.get("issue_types", [])
    if not isinstance(issues, list):
        raise ValueError("issue_types 必须是数组")
    issues = [str(x) for x in issues if str(x) in ALLOWED_ISSUES]
    raw_dialogue = value.get("cleaned_dialogue", [])
    if not isinstance(raw_dialogue, list):
        raise ValueError("cleaned_dialogue 必须是数组")
    if not value["keep"]:
        return {"keep": False, "cleaned_dialogue": [], "issue_types": issues, "reason": reason}

    dialogue: list[dict[str, str]] = []
    for turn in raw_dialogue:
        if not isinstance(turn, dict):
            raise ValueError("cleaned_dialogue 中存在非对象轮次")
        role = normalize_role(turn.get("role"))
        content = str(turn.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            raise ValueError("清洗结果含非法角色或空文本")
        dialogue.append({"role": role, "content": content})
    if len(dialogue) < 2 or {turn["role"] for turn in dialogue} != {"user", "assistant"}:
        raise ValueError("清洗结果不是至少包含医患各一轮的双向对话")
    return {"keep": True, "cleaned_dialogue": dialogue, "issue_types": issues, "reason": reason}


def get_client(api_key: str, base_url: str | None, timeout: float) -> Any:
    client = getattr(_thread_local, "client", None)
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("缺少 openai 包，请先执行：pip install openai") from exc
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        _thread_local.client = client
    return client


def clean_one(
    dialogue: list[dict[str, str]],
    *,
    api_key: str,
    base_url: str | None,
    model: str,
    timeout: float,
    retries: int,
    temperature: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    user_payload = "请清洗以下对话：\n" + json.dumps(dialogue, ensure_ascii=False)
    for attempt in range(retries):
        try:
            client = get_client(api_key, base_url, timeout)
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_payload},
                ],
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            return validate_result(extract_json_object(content))
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(20.0, 2**attempt + random.random()))
    raise RuntimeError(f"LLM 调用或结果校验连续失败 {retries} 次：{last_error}")


def replace_dialogue(record: dict[str, Any], field_path: str, dialogue: list[dict[str, str]]) -> dict[str, Any]:
    result = copy.deepcopy(record)
    if field_path in {"messages", "conversations"}:
        result[field_path] = dialogue
    else:
        speaker_dialogue = [
            {"speaker": "患者" if turn["role"] == "user" else "医生", "content": turn["content"]}
            for turn in dialogue
        ]
        if field_path == "dialogue":
            result["dialogue"] = speaker_dialogue
        elif field_path == "cleared_data.dialogue":
            result.setdefault("cleared_data", {})["dialogue"] = speaker_dialogue
        else:
            result["messages"] = dialogue
    return result


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    problem_path = Path(args.problem_ids_output)
    error_path = Path(args.error_output) if args.error_output else output_path.with_name(
        output_path.stem + "_errors.jsonl"
    )
    records = load_records(input_path)
    if args.limit is not None:
        records = records[: args.limit]

    cleaned_by_index: dict[int, dict[str, Any]] = {}
    problems: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    futures: dict[Future[dict[str, Any]], tuple[int, dict[str, Any], str, str]] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, raw_record in enumerate(records):
            rid = record_id(raw_record, index, args.id_field)
            dialogue, field_path = find_dialogue(raw_record, args.max_turn_chars)
            if len(dialogue) < 2 or {turn["role"] for turn in dialogue} != {"user", "assistant"}:
                problems.append(
                    {"record_id": rid, "input_index": index, "reason": "缺少可识别的双向医患对话"}
                )
                continue
            future = executor.submit(
                clean_one,
                dialogue,
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                retries=args.retries,
                temperature=args.temperature,
            )
            futures[future] = (index, raw_record, rid, field_path)

        completed = 0
        for future in as_completed(futures):
            index, raw_record, rid, field_path = futures[future]
            completed += 1
            try:
                llm_result = future.result()
                if not llm_result["keep"]:
                    problems.append(
                        {
                            "record_id": rid,
                            "input_index": index,
                            "reason": llm_result["reason"] or "LLM 判定为非对话",
                            "issue_types": llm_result["issue_types"],
                        }
                    )
                else:
                    cleaned = replace_dialogue(raw_record, field_path, llm_result["cleaned_dialogue"])
                    cleaned["cleaning_metadata"] = {
                        "issue_types": llm_result["issue_types"],
                        "reason": llm_result["reason"],
                        "model": args.model,
                    }
                    cleaned_by_index[index] = cleaned
            except Exception as exc:
                errors.append(
                    {"record_id": rid, "input_index": index, "error": str(exc), "record": raw_record}
                )
            print(
                f"\r已完成 {completed}/{len(futures)}，保留 {len(cleaned_by_index)}，"
                f"删除 {len(problems)}，错误 {len(errors)}",
                end="",
                flush=True,
            )

    if futures:
        print()
    cleaned_records = [cleaned_by_index[index] for index in sorted(cleaned_by_index)]
    problems.sort(key=lambda item: item["input_index"])
    errors.sort(key=lambda item: item["input_index"])
    write_records(output_path, cleaned_records)
    write_json(problem_path, problems)
    write_records(error_path, errors)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "model": args.model,
        "total": len(records),
        "kept": len(cleaned_records),
        "removed_problem_records": len(problems),
        "errors": len(errors),
        "problem_ids_output": str(problem_path),
        "error_output": str(error_path),
    }
    report_path = output_path.with_name(output_path.stem + "_report.json")
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
