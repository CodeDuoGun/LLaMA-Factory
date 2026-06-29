import argparse
import asyncio
import base64
import json
import mimetypes
import re
import sys
from pathlib import Path
from typing import Any


LABEL_OPTIONS = {
    "tongue_body_color": {"淡红", "红", "暗红", "淡白", "紫暗", "未知"},
    "tongue_shape": {"正常", "胖大", "瘦薄", "齿痕", "裂纹", "未知"},
    "tongue_coating_color": {"白苔", "黄苔", "灰黑苔", "少苔", "无苔", "未知"},
    "tongue_coating_thickness": {"薄苔", "厚苔", "腻苔", "腐苔", "剥脱苔", "未知"},
    "moisture": {"润", "燥", "滑", "未知"},
    "tooth_marks": {"有齿痕", "无齿痕", "不确定"},
    "crack": {"有裂纹", "无裂纹", "不确定"},
}

DEFAULT_LABELS = {
    "tongue_body_color": "未知",
    "tongue_shape": "未知",
    "tongue_coating_color": "未知",
    "tongue_coating_thickness": "未知",
    "moisture": "未知",
    "tooth_marks": "不确定",
    "crack": "不确定",
}

FACE_OPTIONS = {
    "face_complexion": {"正常", "红润", "潮红", "淡白", "萎黄", "晦暗", "青紫", "未知"},
    "face_luster": {"有光泽", "少光泽", "晦暗", "油亮", "未知"},
    "face_oiliness": {"干燥", "正常", "偏油", "明显油腻", "未知"},
    "face_edema": {"无", "轻度", "中度", "重度", "未知"},
    "spirit": {"有神", "神疲", "未知"},
}

DEFAULT_FACE = {
    "face_complexion": "未知",
    "face_luster": "未知",
    "face_oiliness": "未知",
    "face_edema": "未知",
    "spirit": "未知",
}

LESION_OPTIONS = {
    "primary_lesions": {
        "闭合性粉刺",
        "开放性粉刺",
        "丘疹",
        "脓疱",
        "结节",
        "囊肿",
        "红斑",
        "斑疹",
        "丘疱疹",
        "水疱",
        "风团",
        "鳞屑",
        "无明显皮损",
    },
    "secondary_lesions": {
        "痘印",
        "色素沉着",
        "色素减退",
        "瘢痕",
        "抓痕",
        "结痂",
        "糜烂",
        "渗液",
        "苔藓样变",
        "无",
    },
    "lesion_color": {"肤色", "淡红", "鲜红", "暗红", "紫红", "紫暗", "黄", "褐", "黑", "未知"},
    "erythema": {"无", "轻度", "中度", "重度", "未知"},
    "swelling": {"无", "轻度", "中度", "重度", "未知"},
    "shape": {"圆形", "椭圆形", "片状", "斑块状", "融合", "不规则", "未知"},
}

DISTRIBUTION_OPTIONS = {
    "额头",
    "双颊",
    "左颊",
    "右颊",
    "鼻部",
    "鼻翼",
    "口周",
    "下巴",
    "下颌",
    "耳周",
    "颈部",
    "胸部",
    "背部",
    "四肢",
    "手部",
    "腿部",
    "未知",
}

LESION_COUNT_OPTIONS = {"极少", "少量", "中等", "较多", "大量", "未知"}

PROMPT = """你是一名具有30年以上临床经验的中医皮肤科主任医师，同时也是医学图像标注专家。请只根据输入图片判断类型，并分不同类型给出分析结果，输出严格 JSON，不要输出 Markdown 或解释文字。

【分析原则】

1. 只能描述图片中真实可见的信息。
2. 不允许猜测患者疾病。
3. 不允许进行中医辨证。
4. 不允许推测病因。
5. 不允许根据经验补充图片中看不到的信息。
6. 如果某项无法判断，请填写"未知"。
7. 不要输出任何解释。
8. 仅输出JSON。

面部照片/患处照片输出内容：
- face_complexion（面色）
    可选值：
    正常、红润、潮红、淡白、萎黄、晦暗、青紫、未知
- face_luster（光泽）
    可选值：
    有光泽、少光泽、晦暗、油亮、未知
- face_oiliness（皮肤油脂）
    可选值：
    干燥、正常、偏油、明显油腻、未知
- face_edema（浮肿）
    可选值：
    无、轻度、中度、重度、未知
- spirit（神色）
    可选值：
    有神、神疲、未知
适用范围：面部照片，或患处照片，包括正面、侧面、手部、背部、腿部等有皮肤问题的照片。
皮损情况（lesion）
- primary_lesions（原发皮损）
可多选：
闭合性粉刺、开放性粉刺、丘疹、脓疱、结节、囊肿、红斑、斑疹、丘疱疹、水疱、风团、鳞屑、无明显皮损
- secondary_lesions（继发皮损）
可多选：
痘印、色素沉着、色素减退、瘢痕、抓痕、结痂、糜烂、渗液、苔藓样变、无

- lesion_color（皮损颜色）
可选值：
肤色、淡红、鲜红、暗红、紫红、紫暗、黄、褐、黑、未知
四、炎症程度
- erythema（红斑）
可选值：
无、轻度、中度、重度、未知

- swelling（肿胀）
可选值：
无、轻度、中度、重度、未知

- lesion_distribution（皮损分布）
可多选：额头、双颊、左颊、右颊、鼻部、鼻翼、口周、下巴、下颌、耳周、颈部、胸部、背部、四肢、手部、腿部、未知

- lesion_count（皮损数量）
可选值：
极少、少量、中等、较多、大量、未知

- shape（形态）
可选值：圆形、椭圆形、片状、斑块状、融合、不规则、未知
最终输出格式：
{
    "image_type":"face",
    "face":{
        "face_complexion":"",
        "face_luster":"",
        "face_oiliness":"",
        "face_edema":"",
        "spirit":""
    },
    "lesion":{
        "primary_lesions":[],
        "secondary_lesions":[],
        "lesion_color":"",
        "erythema":"",
        "swelling":"",
        "shape":""
    },
    "distribution":[],
    "lesion_count":"",
    "description":""
}


舌象照片输出内容：适用范围包括舌上照片和舌下照片。
输出格式必须为：
{
    "image_type":"tongue",
    "labels": {
        "tongue_body_color": "淡红/红/暗红/淡白/紫暗/未知",
        "tongue_shape": "正常/胖大/瘦薄/齿痕/裂纹/未知",
        "tongue_coating_color": "白苔/黄苔/灰黑苔/少苔/无苔/未知",
        "tongue_coating_thickness": "薄苔/厚苔/腻苔/腐苔/剥脱苔/未知",
        "moisture": "润/燥/滑/未知",
        "tooth_marks": "有齿痕/无齿痕/不确定",
        "crack": "有裂纹/无裂纹/不确定"
    },
  "description": "一句中文舌象描述"
}

报告照片输出内容：
report_type: 报告类型，如：血常规、尿常规、肝功能、肾功能、血糖、血脂、甲状腺功能、肿瘤标志物、心电图、胸片、腹部超声、骨密度、眼底检查、住院报告、体检报告等其他类型。无法确认给“未知”
description: 报告内容，有无异常指标，若有异常指标，简单解释。
最终输出格式：
{
    "image_type":"report",
    "report":{
        "report_type":"",
        "description":""
    }
}

不属于以上类型的图片，或无法判断为舌照、面部/患处、报告时，输出：
{
    "image_type":"invalid"
}

要求：
1. 每个标签只能从给定枚举中选择一个值。
2. 看不清或无法判断时使用“未知”或“不确定”。
3. description 要综合所有可判断特征，保持简洁。
"""


def build_image_url(image_path: str) -> str:
    if image_path.startswith(("http://", "https://", "data:image/")):
        return image_path

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    image_base64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def build_messages(image_path: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": build_image_url(image_path)}},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("未找到 JSON 对象") from None
        return json.loads(match.group(0))


def normalize_choice(value: Any, options: set[str], default_value: str = "未知") -> str:
    return value if isinstance(value, str) and value in options else default_value


def normalize_multi_choice(value: Any, options: set[str], default_value: str = "未知") -> list[str]:
    values = value if isinstance(value, list) else [value]
    normalized = [item for item in values if isinstance(item, str) and item in options]
    return normalized or [default_value]


def normalize_result(raw: dict[str, Any], image_path: str, image_id: str | None = None) -> dict[str, Any]:
    image_type = raw.get("image_type")
    if image_type == "tongue" or (image_type is None and "labels" in raw):
        return normalize_tongue_result(raw, image_path=image_path, image_id=image_id)
    if image_type == "face":
        return normalize_face_result(raw, image_path=image_path, image_id=image_id)
    if image_type == "report":
        return normalize_report_result(raw, image_path=image_path, image_id=image_id)
    return normalize_invalid_result(image_path=image_path, image_id=image_id)


def base_result(image_path: str, image_id: str | None, image_type: str) -> dict[str, Any]:
    path = Path(image_path)
    return {
        "id": image_id or path.stem,
        "image": image_path,
        "image_type": image_type,
    }


def normalize_tongue_result(raw: dict[str, Any], image_path: str, image_id: str | None = None) -> dict[str, Any]:
    labels = raw.get("labels") if isinstance(raw.get("labels"), dict) else raw
    normalized_labels = {}
    for key, default_value in DEFAULT_LABELS.items():
        value = labels.get(key, default_value) if isinstance(labels, dict) else default_value
        normalized_labels[key] = normalize_choice(value, LABEL_OPTIONS[key], default_value)

    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        description = build_description(normalized_labels)

    result = base_result(image_path, image_id, "tongue")
    result.update({"labels": normalized_labels, "description": description.strip()})
    return result


def normalize_face_result(raw: dict[str, Any], image_path: str, image_id: str | None = None) -> dict[str, Any]:
    face = raw.get("face") if isinstance(raw.get("face"), dict) else {}
    lesion = raw.get("lesion") if isinstance(raw.get("lesion"), dict) else {}

    normalized_face = {
        key: normalize_choice(face.get(key), FACE_OPTIONS[key], default_value)
        for key, default_value in DEFAULT_FACE.items()
    }
    normalized_lesion = {
        "primary_lesions": normalize_multi_choice(
            lesion.get("primary_lesions"), LESION_OPTIONS["primary_lesions"], "无明显皮损"
        ),
        "secondary_lesions": normalize_multi_choice(lesion.get("secondary_lesions"), LESION_OPTIONS["secondary_lesions"], "无"),
        "lesion_color": normalize_choice(lesion.get("lesion_color"), LESION_OPTIONS["lesion_color"]),
        "erythema": normalize_choice(lesion.get("erythema"), LESION_OPTIONS["erythema"]),
        "swelling": normalize_choice(lesion.get("swelling"), LESION_OPTIONS["swelling"]),
        "shape": normalize_choice(lesion.get("shape"), LESION_OPTIONS["shape"]),
    }

    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        description = build_face_description(normalized_face, normalized_lesion)

    result = base_result(image_path, image_id, "face")
    result.update(
        {
            "face": normalized_face,
            "lesion": normalized_lesion,
            "distribution": normalize_multi_choice(raw.get("distribution"), DISTRIBUTION_OPTIONS),
            "lesion_count": normalize_choice(raw.get("lesion_count"), LESION_COUNT_OPTIONS),
            "description": description.strip(),
        }
    )
    return result


def normalize_report_result(raw: dict[str, Any], image_path: str, image_id: str | None = None) -> dict[str, Any]:
    report = raw.get("report") if isinstance(raw.get("report"), dict) else {}
    report_type = report.get("report_type")
    description = report.get("description") or raw.get("description")

    result = base_result(image_path, image_id, "report")
    result["report"] = {
        "report_type": report_type.strip() if isinstance(report_type, str) and report_type.strip() else "未知",
        "description": description.strip() if isinstance(description, str) and description.strip() else "未知",
    }
    return result


def normalize_invalid_result(image_path: str, image_id: str | None = None) -> dict[str, Any]:
    return base_result(image_path, image_id, "invalid")


def build_description(labels: dict[str, str]) -> str:
    parts = []
    if labels["tongue_body_color"] != "未知":
        parts.append(f"舌质{labels['tongue_body_color']}")
    if labels["tongue_shape"] != "未知":
        parts.append(f"舌体{labels['tongue_shape']}")
    if labels["tooth_marks"] == "有齿痕":
        parts.append("边有齿痕")
    elif labels["tooth_marks"] == "无齿痕":
        parts.append("无齿痕")
    if labels["crack"] == "有裂纹":
        parts.append("有裂纹")
    elif labels["crack"] == "无裂纹":
        parts.append("无裂纹")

    coating = []
    if labels["tongue_coating_color"] != "未知":
        coating.append(labels["tongue_coating_color"])
    if labels["tongue_coating_thickness"] != "未知":
        coating.append(labels["tongue_coating_thickness"])
    if labels["moisture"] != "未知":
        coating.append(labels["moisture"])
    if coating:
        parts.append("苔" + "而".join(coating))

    return "，".join(parts) + "。" if parts else "舌象特征不清，无法确定。"


def build_face_description(face: dict[str, str], lesion: dict[str, Any]) -> str:
    parts = []
    if face["face_complexion"] != "未知":
        parts.append(f"面色{face['face_complexion']}")
    if face["face_luster"] != "未知":
        parts.append(face["face_luster"])
    if face["face_oiliness"] != "未知":
        parts.append(f"皮肤{face['face_oiliness']}")
    if lesion["primary_lesions"] != ["无明显皮损"]:
        parts.append("可见" + "、".join(lesion["primary_lesions"]))
    if lesion["secondary_lesions"] != ["无"]:
        parts.append("伴" + "、".join(lesion["secondary_lesions"]))
    if lesion["erythema"] != "未知":
        parts.append(f"红斑{lesion['erythema']}")
    if lesion["swelling"] != "未知":
        parts.append(f"肿胀{lesion['swelling']}")
    return "，".join(parts) + "。" if parts else "皮肤图像特征不清，无法确定。"


async def call_qwen_vision(image_path: str) -> str:
    from openai import AsyncOpenAI

    from medical.config import config

    client = AsyncOpenAI(api_key=config.DASHSCOPE_API_KEY, base_url=config.DASHSCOPE_BASE_URL, timeout=(5.0, 60))
    response = await client.chat.completions.create(
        model=config.QWEN_IMAGE_DESCRIBE_MODEL,
        messages=build_messages(image_path),
        temperature=0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


async def analyze_image(image_path: str, image_id: str | None = None) -> dict[str, Any]:
    content = await call_qwen_vision(image_path)
    raw = extract_json_object(content)
    return normalize_result(raw, image_path=image_path, image_id=image_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用视觉大模型分析医学图片，输出 tongue/face/report/invalid 结构化 JSON。")
    parser.add_argument("image", nargs="?", help="单张图片路径、图片 URL 或 data:image URL")
    parser.add_argument("--id", dest="image_id", default=None, help="单张图片输出 id，默认使用文件名 stem")
    parser.add_argument("--input-dir", default=None, help="批量分析目录中的图片")
    parser.add_argument("--glob", default="*.jpg", help="批量分析文件匹配规则，默认 *.jpg")
    parser.add_argument("--output", "-o", default=None, help="输出文件；单张为 JSON，批量为 JSONL；不传则打印")
    return parser.parse_args()


async def analyze_batch(input_dir: str, pattern: str) -> list[dict[str, Any]]:
    image_paths = sorted(Path(input_dir).glob(pattern))
    results = []
    for image_path in image_paths:
        try:
            results.append(await analyze_image(str(image_path), image_id=image_path.stem))
        except Exception as exc:
            print(f"跳过处理失败的图片: {image_path}，原因: {exc}", file=sys.stderr)
            continue
    return results


def main() -> None:
    args = parse_args()
    if args.input_dir:
        results = asyncio.run(analyze_batch(args.input_dir, args.glob))
        output = "\n".join(json.dumps(result, ensure_ascii=False) for result in results)
    elif args.image:
        result = asyncio.run(analyze_image(args.image, image_id=args.image_id))
        output = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        raise SystemExit("请传入 image 或 --input-dir")

    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
