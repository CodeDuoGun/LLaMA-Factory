import argparse
import base64
import json
import mimetypes
import re
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

PROMPT = """你是一名中医舌象图像标注助手。请只根据输入图片判断舌象，输出严格 JSON，不要输出 Markdown 或解释文字。

输出格式必须为：
{
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


def normalize_result(raw: dict[str, Any], image_path: str, image_id: str | None = None) -> dict[str, Any]:
    labels = raw.get("labels") if isinstance(raw.get("labels"), dict) else raw
    normalized_labels = {}
    for key, default_value in DEFAULT_LABELS.items():
        value = labels.get(key, default_value) if isinstance(labels, dict) else default_value
        normalized_labels[key] = value if value in LABEL_OPTIONS[key] else default_value

    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        description = build_description(normalized_labels)

    path = Path(image_path)
    return {
        "id": image_id or path.stem,
        "image": image_path,
        "image_type": "tongue",
        "labels": normalized_labels,
        "description": description.strip(),
    }


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


def call_qwen_vision(image_path: str) -> str:
    from openai import OpenAI

    from medical.config import config

    client = OpenAI(api_key=config.DASHSCOPE_API_KEY, base_url=config.DASHSCOPE_BASE_URL, timeout=(5.0, 60))
    response = client.chat.completions.create(
        model=config.QWEN_IMAGE_DESCRIBE_MODEL,
        messages=build_messages(image_path),
        temperature=0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def analyze_image(image_path: str, image_id: str | None = None) -> dict[str, Any]:
    content = call_qwen_vision(image_path)
    raw = extract_json_object(content)
    return normalize_result(raw, image_path=image_path, image_id=image_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用视觉大模型分析舌象图片并输出结构化 JSON。")
    parser.add_argument("image", help="图片路径、图片 URL 或 data:image URL")
    parser.add_argument("--id", dest="image_id", default=None, help="输出 JSON 的 id，默认使用图片文件名 stem")
    parser.add_argument("--output", "-o", default=None, help="结果保存路径；不传则打印到 stdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_image(args.image, image_id=args.image_id)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
