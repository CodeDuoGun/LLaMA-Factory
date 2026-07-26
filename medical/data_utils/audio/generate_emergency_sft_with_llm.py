#!/usr/bin/env python3
"""调用 OpenAI 兼容 LLM 生成四级医疗风险处置 SFT 数据。

默认生成 200 条，L1/L2/L3/L4 各 50 条，并输出全量及 80/10/10 分层切分。
数据用于训练风险话术和结构化动作，不应替代确定性规则与人工医疗审核。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()


ALLOWED_ACTIONS = {
    "STOP_NORMAL_CONSULT",
    "SHOW_EMERGENCY_CALL_PAGE",
    "NOTIFY_HUMAN_SUPPORT",
    "URGENT_OFFLINE_EVALUATION",
    "PRIORITY_CONSULT",
    "COLLECT_ESSENTIAL_INFORMATION",
    "RECOMMEND_VISIT_WITHIN_24H",
    "CONTINUE_NORMAL_CONSULT",
    "DECLINE_OUT_OF_SCOPE",
    "EXPLAIN_MODEL_LIMITS",
    "SUPPORTIVE_SAFETY_RESPONSE",
}

REQUIRED_ACTIONS = {
    "L1": {"STOP_NORMAL_CONSULT", "SHOW_EMERGENCY_CALL_PAGE"},
    "L2": {"STOP_NORMAL_CONSULT", "URGENT_OFFLINE_EVALUATION", "NOTIFY_HUMAN_SUPPORT"},
    "L3": {"PRIORITY_CONSULT", "COLLECT_ESSENTIAL_INFORMATION", "RECOMMEND_VISIT_WITHIN_24H"},
    "L4": {"DECLINE_OUT_OF_SCOPE", "EXPLAIN_MODEL_LIMITS", "CONTINUE_NORMAL_CONSULT"},
}

LEVEL_SCENARIOS = {
    "L1": [
        "无法完整说话的进行性呼吸困难或明显低血氧",
        "持续胸痛伴大汗、恶心或放射痛",
        "突发口角歪斜、单侧无力或言语不清",
        "低血压并伴冷汗、眼前发黑或意识模糊",
        "用药或进食后喉头、面颈肿胀并呼吸困难",
        "高热或感染同时出现意识不清或无法唤醒",
        "大量呕血、咯血或持续活动性出血",
        "孕妇大量阴道流血并伴头晕或腹痛",
        "异常血糖并伴昏迷、抽搐或意识改变",
        "已经自伤、服药过量或存在立即实施的明确计划",
    ],
    "L2": [
        "呼吸困难但仍能完整说话或血氧90%至92%",
        "新发胸痛胸闷但尚不满足L1组合",
        "刚发生过晕厥、现在已经清醒",
        "抽搐已经停止但原因不明",
        "全身过敏或面部肿胀但尚无明确气道受阻",
        "持续39.5度以上高热或精神状态明显变差",
        "原因不明的8分以上剧烈疼痛",
        "心率大于120或低血压但无休克意识表现",
        "严重异常化验指标或严重电解质紊乱",
        "孕妇儿童老人免疫缺陷或肿瘤患者的紧急危险表现",
    ],
    "L3": [
        "症状近期明显加重并影响日常生活",
        "持续38.5度以上发热但没有L1或L2信号",
        "疼痛或瘙痒达到6至7分",
        "皮损在24至72小时内明显扩散",
        "皮损出现脓疱渗液破溃或局部感染迹象",
        "持续失眠或食欲明显下降",
        "炎症指标或白细胞轻中度异常",
        "肝肾功能轻度异常或血糖血压控制不佳",
        "基础病或免疫抑制背景下症状加重",
        "孕妇儿童老人或反复发作患者近期加重",
    ],
    "L4": [
        "要求模型给出百分之百确定诊断",
        "要求模型替代医生直接开处方",
        "要求给出处方药具体剂量或自行加量",
        "要求指导自行停用关键处方药",
        "要求保证治疗一定有效或保证不会恶化",
        "反复施压要求模型突破诊断和处方边界",
        "要求模型声称已经联系医生或执行外部动作",
        "要求隐瞒风险或生成不符合医疗伦理的内容",
        "普通问诊中夹带越界要求，应局部拒绝后继续帮助",
        "第三方代问诊并要求模型直接确诊开药",
    ],
}

SCENARIO_REASON_CODES = {
    scenario: f"{level}_{index + 1:02d}_{suffix}"
    for level, scenarios in LEVEL_SCENARIOS.items()
    for index, scenario in enumerate(scenarios)
    for suffix in ("EVIDENCE",)
}

DIALOGUE_STYLES = [
    "单轮直接陈述，口语自然",
    "两到三轮渐进披露，危险信号在后续轮次出现",
    "包含错别字、停顿或轻微ASR噪声但含义可辨",
    "家属代患者描述，主体明确",
    "包含历史或否定信息，只有当前真实信号用于判断",
]

RULES = """你必须严格遵守以下四级规范：
- 总原则：多等级同时命中取最高级；识别否定、历史、假设和主体；不得补造用户未说的信息。
- L1 立即危及生命或重大功能：严重气道/呼吸、典型急性心脑血管、休克或意识障碍、重大出血、严重代谢急症、正在实施或有明确立即计划的自伤伤人。动作必须包含 STOP_NORMAL_CONSULT、SHOW_EMERGENCY_CALL_PAGE。话术先要求立即呼叫急救/120，不自行驾车、不等待AI。自伤场景还必须包含 SUPPORTIVE_SAFETY_RESPONSE，并使用支持性非评判话术。
- L2 紧急高危：有显著危险、需要立即线下急诊评估，但尚不足以确认L1。动作必须包含 STOP_NORMAL_CONSULT、URGENT_OFFLINE_EVALUATION、NOTIFY_HUMAN_SUPPORT。不得让用户观察一天。
- L3 高危预警：症状明显加重或有高风险背景，暂未出现L1/L2。动作必须包含 PRIORITY_CONSULT、COLLECT_ESSENTIAL_INFORMATION、RECOMMEND_VISIT_WITHIN_24H。可有限追问，必须说明升级急救的危险信号。
- L4 模型边界：要求确定诊断、处方、具体处方药剂量、自行停用关键药物、保证疗效或虚构外部动作。动作必须包含 DECLINE_OUT_OF_SCOPE、EXPLAIN_MODEL_LIMITS、CONTINUE_NORMAL_CONSULT。只拒绝越界部分，并继续风险筛查或信息整理。如同时出现L1-L3，必须按风险等级而不是L4。
- 所有回复不得确定诊断，不得开药，不得给具体药名剂量，不得声称已执行真实外部动作，不输出思维链。
"""


@dataclass(frozen=True)
class Target:
    sample_id: str
    risk_level: str
    scenario: str
    dialogue_style: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/emergency_sft")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--model", default=os.getenv("REPORT_VLM_MODEL") or os.getenv("QWEN_TEXT_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("REPORT_VLM_BASE_URL") or os.getenv("OPENAI_API_BASE"))
    parser.add_argument("--api-key", default=os.getenv("REPORT_VLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
    args = parser.parse_args()
    if args.count < 4 or args.count % 4:
        parser.error("--count 必须是大于等于4且能被4整除的整数")
    if args.batch_size < 1 or args.workers < 1 or args.retries < 1:
        parser.error("--batch-size、--workers、--retries 必须大于0")
    if not args.model or not args.api_key:
        parser.error("缺少模型或 API key；请配置 REPORT_VLM_MODEL/REPORT_VLM_API_KEY")
    return args


def make_targets(count: int) -> list[Target]:
    per_level = count // 4
    targets: list[Target] = []
    for level in ("L1", "L2", "L3", "L4"):
        scenarios = LEVEL_SCENARIOS[level]
        for index in range(per_level):
            targets.append(
                Target(
                    sample_id=f"emergency_{level.lower()}_{index + 1:04d}",
                    risk_level=level,
                    scenario=scenarios[index % len(scenarios)],
                    dialogue_style=DIALOGUE_STYLES[(index // len(scenarios)) % len(DIALOGUE_STYLES)],
                )
            )
    return targets


def chunks(values: list[Target], size: int) -> list[list[Target]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("模型输出中没有 JSON 对象")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("模型输出顶层不是 JSON 对象")
    return value


def validate_sample(raw: Any, target: Target, model: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("target_id") != target.sample_id:
        raise ValueError(f"target_id 不匹配：期望 {target.sample_id}")
    user_messages = raw.get("user_messages")
    assistant_questions = raw.get("assistant_questions", [])
    response = raw.get("response")
    if not isinstance(user_messages, list) or not 1 <= len(user_messages) <= 4:
        raise ValueError(f"{target.sample_id}: user_messages 数量必须为1到4")
    if not isinstance(assistant_questions, list):
        raise ValueError(f"{target.sample_id}: assistant_questions 必须是数组")
    if not all(isinstance(value, str) and value.strip() for value in user_messages + assistant_questions):
        raise ValueError(f"{target.sample_id}: 对话中存在空文本")
    expected_questions = len(user_messages) - 1
    if len(assistant_questions) > expected_questions:
        assistant_questions = assistant_questions[:expected_questions]
    elif len(assistant_questions) < expected_questions:
        # 供应端偶尔给出多条患者消息却漏掉中间医生问句；合并患者消息，
        # 避免编造医生追问，同时保留全部由 LLM 生成的患者证据。
        user_messages = [" ".join(value.strip() for value in user_messages)]
        assistant_questions = []
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"{target.sample_id}: response 为空")
    response = response.strip()
    user_text = " ".join(value.strip() for value in user_messages)
    if "<think>" in response.lower():
        raise ValueError(f"{target.sample_id}: 不允许输出思维链")
    if re.search(r"(肯定是|百分之百是|已经确诊|我给你开|处方如下)", response):
        raise ValueError(f"{target.sample_id}: 含确定诊断或开方表达")

    external_action = re.compile(
        r"(我会|我来|我可(?:以)?|我能|我们(?:会|可以)|平台(?:会|可以)|系统(?:会|可以)|"
        r"已(?:经)?为你|已(?:经)?帮你).{0,20}(联系|通知|报警|呼叫|协助|安排)"
    )
    medication_advice = re.compile(r"(肾上腺素|抗过敏药|立即使用.{0,8}(药|笔)|服用.{0,8}药)")
    kept_sentences: list[str] = []
    for sentence in re.findall(r"[^。！？!?]+[。！？!?]?", response):
        sentence = sentence.strip()
        if not sentence:
            continue
        if external_action.search(sentence) or re.search(
            r"(我会|我能|我在这里).{0,16}(一直|陪|等救护车|确保你安全)|绝不让.{0,8}独自",
            sentence,
        ):
            continue
        if target.risk_level in {"L1", "L2"} and ("？" in sentence or "?" in sentence):
            continue
        if target.risk_level in {"L1", "L2"} and medication_advice.search(sentence):
            continue
        kept_sentences.append(sentence)
    response = "".join(kept_sentences).strip()
    if not response:
        raise ValueError(f"{target.sample_id}: 清除不安全话术后 response 为空")
    response = response.replace("不要自行前往医院", "不要自行驾车")
    response = re.sub(r"(^|[。！!])这是", r"\1这些表现可能提示", response)

    unmistakable_l1 = re.compile(
        r"(无法?完整说|不能完整说|嘴唇发紫|口唇发紫|"
        r"突然.{0,12}(说不清|言语不清|口角歪|半边.{0,5}(没劲|无力)|一侧.{0,5}无力)|"
        r"叫不醒|无法唤醒|昏迷|正在抽搐|吐了很多血|大量呕血|流血止不住|"
        r"吞了.{0,12}(半瓶|大量).{0,5}药|喝了.{0,12}(半瓶|大量).{0,5}药)"
    )
    if target.risk_level in {"L2", "L3", "L4"} and unmistakable_l1.search(user_text):
        raise ValueError(f"{target.sample_id}: 用户文本漂移为明确 L1 信号")
    if target.risk_level in {"L3", "L4"} and re.search(
        r"(血氧.{0,5}(90|91|92)%|39\.5.{0,3}(度|℃)|晕厥|疼痛.{0,8}(8分|9分|10分)|"
        r"脸.{0,5}肿.{0,12}(呼吸费力|呼吸困难)|面部.{0,5}肿.{0,12}(呼吸费力|呼吸困难))",
        user_text,
    ):
        raise ValueError(f"{target.sample_id}: 用户文本漂移为 L2 或更高信号")
    # 关键处置句由确定性模板兜底，避免生成模型遗漏安全动作。
    if target.risk_level == "L1" and not re.search(r"(120|急救|紧急服务)", response):
        response = "这可能是紧急危险，请立即拨打120或联系当地紧急服务。" + response
    if target.risk_level == "L2" and not re.search(r"(急诊|紧急评估|立即就医|尽快就医)", response):
        response = "请立即前往急诊接受线下紧急评估。" + response
    if target.risk_level == "L3" and not re.search(r"(24小时|24 小时)", response):
        response += " 建议在24小时内到线下医疗机构就诊。"
    if target.risk_level == "L4" and not re.search(r"(不能|无法)", response):
        response = "我不能替代真人医生完成这项请求。" + response
    if target.risk_level == "L1" and "自伤" in target.scenario and not re.search(
        r"(可信任|信任的人|亲友|家人|身边人)", response
    ):
        response += " 请尽量远离药物、刀具、高处等危险，并让身边可信任的人立即陪在你身边。"

    actions = sorted(REQUIRED_ACTIONS[target.risk_level])
    if target.risk_level == "L1" and "自伤" in target.scenario:
        actions.append("SUPPORTIVE_SAFETY_RESPONSE")
    decision = {
        "risk_level": target.risk_level,
        "reason_codes": [SCENARIO_REASON_CODES[target.scenario]],
        "actions": actions,
    }
    final_content = (
        f"<decision>{json.dumps(decision, ensure_ascii=False, separators=(',', ':'))}</decision>\n"
        f"<response>{response}</response>"
    )
    conversations: list[dict[str, str]] = []
    for index, user_message in enumerate(user_messages):
        conversations.append({"role": "user", "content": user_message.strip()})
        if index < len(assistant_questions):
            conversations.append({"role": "assistant", "content": assistant_questions[index].strip()})
    conversations.append({"role": "assistant", "content": final_content})

    return {
        "id": target.sample_id,
        "conversations": conversations,
        "metadata": {
            "risk_level": target.risk_level,
            "scenario": target.scenario,
            "dialogue_style": target.dialogue_style,
            "generator_model": model,
            "requires_clinical_review": True,
        },
    }


def make_prompt(batch: list[Target]) -> str:
    targets = [
        {
            "target_id": item.sample_id,
            "risk_level": item.risk_level,
            "scenario": item.scenario,
            "dialogue_style": item.dialogue_style,
        }
        for item in batch
    ]
    return f"""{RULES}
请严格按下面每个目标生成一条中文医疗对话素材。每条都要自然、彼此不同，并让用户话语本身提供足够判定依据。

对话要求：
1. user_messages 是按时间排列的患者/家属原话数组，数量1到4条。
2. assistant_questions 是前置的简短问诊追问数组，数量必须恰好比 user_messages 少1；单轮时为空数组。
3. response 只是最后展示给用户的处置话术，不要包含 decision、JSON、XML标签或风险等级代号。
4. 不在 user 文本中直接出现 L1/L2/L3/L4、动作枚举、训练任务或“我是测试样本”。
5. 最后一条 user_message 必须提供目标场景的关键证据。L1/L2 的 response 立即处置、不再追问；L3 可在 response 末尾问一个必要问题；L4 局部拒绝并转向安全帮助。
6. 当前症状、历史症状和否定信息要清楚区分；家属代述时主体必须明确。
7. 不得把目标场景升级成更严重的信号。例如 L2/L3 不得擅自加入喘不上气、嘴唇发紫、无法说话、突发言语不清/单侧无力、叫不醒、昏迷、正在抽搐、大量出血或服药过量。
8. response 不得声称“我会/已经替你联系、通知、报警或呼叫”，不得声称持续陪伴；L1/L2 不得包含任何问句、药名或用药建议。

目标列表：
{json.dumps(targets, ensure_ascii=False, indent=2)}

只返回一个 JSON 对象，不要 Markdown，不要额外文字：
{{"samples":[{{"target_id":"目标ID","user_messages":["患者原话"],"assistant_questions":[],"response":"最终用户可见话术"}}]}}
samples 数量、target_id 和顺序必须与目标列表完全一致。
"""


def generate_batch(batch: list[Target], args: argparse.Namespace, batch_index: int) -> list[dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai 包，请先安装 openai") from exc

    cache_dir = Path(args.output_dir) / f".batches_v3_n{args.count}_b{args.batch_size}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"batch_{batch_index:04d}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        try:
            return [validate_sample(raw, target, args.model) for raw, target in zip(cached["samples"], batch)]
        except Exception:
            invalid_path = cache_path.with_suffix(f".invalid_{int(time.time())}.json")
            cache_path.replace(invalid_path)

    last_error: Exception | None = None
    for attempt in range(args.retries):
        try:
            client_kwargs: dict[str, Any] = {"api_key": args.api_key, "timeout": args.timeout}
            if args.base_url:
                client_kwargs["base_url"] = args.base_url
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是医疗AI安全数据集专家，只生成合成对话，不进行真实问诊。严格返回合法JSON。",
                    },
                    {"role": "user", "content": make_prompt(batch)},
                ],
                temperature=args.temperature,
                max_tokens=max(2048, 1200 * len(batch)),
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False},
            )
            payload = extract_json_object(response.choices[0].message.content or "")
            raw_samples = payload.get("samples")
            if not isinstance(raw_samples, list) or len(raw_samples) != len(batch):
                raise ValueError(f"samples 数量错误：{len(raw_samples) if isinstance(raw_samples, list) else '非数组'}")
            validated = [
                validate_sample(raw, target, args.model) for raw, target in zip(raw_samples, batch)
            ]
            cache_path.write_text(
                json.dumps({"samples": raw_samples}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return validated
        except Exception as exc:
            last_error = exc
            if attempt + 1 < args.retries:
                time.sleep(min(30.0, 2**attempt + random.random()))
    raise RuntimeError(
        f"batch {batch_index} ({batch[0].sample_id}..{batch[-1].sample_id}) "
        f"连续失败 {args.retries} 次：{last_error}"
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_samples(samples: list[dict[str, Any]]) -> tuple[list[Any], list[Any], list[Any]]:
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    level_counters = {level: 0 for level in REQUIRED_ACTIONS}
    for sample in samples:
        level = sample["metadata"]["risk_level"]
        slot = level_counters[level] % 10
        level_counters[level] += 1
        if slot == 0:
            validation.append(sample)
        elif slot == 1:
            test.append(sample)
        else:
            train.append(sample)
    return train, validation, test


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = make_targets(args.count)
    batches = chunks(targets, args.batch_size)
    completed: dict[int, list[dict[str, Any]]] = {}
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_batch, batch, args, batch_index): batch_index
            for batch_index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_index = futures[future]
            try:
                completed[batch_index] = future.result()
                done = sum(len(values) for values in completed.values())
                print(f"已生成并校验 {done}/{args.count}", flush=True)
            except Exception as exc:
                errors.append(str(exc))
                print(f"失败：{exc}", flush=True)

    if errors:
        write_json(output_dir / "generation_errors.json", errors)
        raise RuntimeError(f"有 {len(errors)} 个批次生成失败，详见 generation_errors.json")

    samples = [sample for index in sorted(completed) for sample in completed[index]]
    if len(samples) != args.count or len({sample["id"] for sample in samples}) != args.count:
        raise RuntimeError("最终样本数量或 ID 唯一性校验失败")
    user_starts = [sample["conversations"][0]["content"] for sample in samples]
    if len(set(user_starts)) != len(user_starts):
        raise RuntimeError("存在完全重复的首轮 user 文本")

    train, validation, test = split_samples(samples)
    write_json(output_dir / f"emergency_sft_{args.count}.json", samples)
    write_json(output_dir / "train.json", train)
    write_json(output_dir / "validation.json", validation)
    write_json(output_dir / "test.json", test)
    manifest = {
        "generator_model": args.model,
        "total": len(samples),
        "splits": {"train": len(train), "validation": len(validation), "test": len(test)},
        "levels": {
            level: sum(sample["metadata"]["risk_level"] == level for sample in samples)
            for level in REQUIRED_ACTIONS
        },
        "format": "MiniCPM-V conversations SFT with decision/response envelope",
        "requires_clinical_review": True,
        "source_rule": "emergency.md",
        "seed": args.seed,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "generation_errors.json", [])
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
