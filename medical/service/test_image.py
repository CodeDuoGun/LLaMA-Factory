import asyncio
import base64

import pytest

from medical.service import image


def test_extract_json_object_from_fenced_response():
    content = """```json
{
  "labels": {
    "tongue_body_color": "淡红",
    "tongue_shape": "胖大",
    "tongue_coating_color": "白苔",
    "tongue_coating_thickness": "厚苔",
    "moisture": "润",
    "tooth_marks": "有齿痕",
    "crack": "无裂纹"
  },
  "description": "舌质淡红，舌体胖大。"
}
```"""

    assert image.extract_json_object(content)["labels"]["tongue_shape"] == "胖大"


def test_normalize_result_fills_missing_and_invalid_labels():
    raw = {
        "labels": {
            "tongue_body_color": "青色",
            "tongue_shape": "胖大",
            "tooth_marks": "有齿痕",
        }
    }

    result = image.normalize_result(raw, image_path="images/000001.jpg", image_id="000001")

    assert result == {
        "id": "000001",
        "image": "images/000001.jpg",
        "image_type": "tongue",
        "labels": {
            "tongue_body_color": "未知",
            "tongue_shape": "胖大",
            "tongue_coating_color": "未知",
            "tongue_coating_thickness": "未知",
            "moisture": "未知",
            "tooth_marks": "有齿痕",
            "crack": "不确定",
        },
        "description": "舌体胖大，边有齿痕。",
    }


def test_normalize_face_result_keeps_only_valid_options():
    raw = {
        "image_type": "face",
        "face": {
            "face_complexion": "潮红",
            "face_luster": "特别亮",
            "face_oiliness": "偏油",
            "face_edema": "无",
            "spirit": "有神",
        },
        "lesion": {
            "primary_lesions": ["丘疹", "不存在的皮损"],
            "secondary_lesions": ["痘印"],
            "lesion_color": "鲜红",
            "erythema": "中度",
            "swelling": "没有",
            "shape": "片状",
        },
        "distribution": ["双颊", "未知部位"],
        "lesion_count": "较多",
        "description": "双颊可见丘疹和痘印，颜色鲜红。",
    }

    result = image.normalize_result(raw, image_path="images/face.jpg", image_id="face001")

    assert result == {
        "id": "face001",
        "image": "images/face.jpg",
        "image_type": "face",
        "face": {
            "face_complexion": "潮红",
            "face_luster": "未知",
            "face_oiliness": "偏油",
            "face_edema": "无",
            "spirit": "有神",
        },
        "lesion": {
            "primary_lesions": ["丘疹"],
            "secondary_lesions": ["痘印"],
            "lesion_color": "鲜红",
            "erythema": "中度",
            "swelling": "未知",
            "shape": "片状",
        },
        "distribution": ["双颊"],
        "lesion_count": "较多",
        "description": "双颊可见丘疹和痘印，颜色鲜红。",
    }


def test_normalize_report_result_fills_missing_report_fields():
    raw = {"image_type": "report", "report": {"report_type": "血常规"}}

    result = image.normalize_result(raw, image_path="images/report.png", image_id="report001")

    assert result == {
        "id": "report001",
        "image": "images/report.png",
        "image_type": "report",
        "report": {
            "report_type": "血常规",
            "description": "未知",
        },
    }


def test_normalize_unknown_type_returns_invalid_only():
    raw = {"image_type": "food", "labels": {"tongue_body_color": "红"}, "description": "一张食物照片"}

    result = image.normalize_result(raw, image_path="images/food.jpg", image_id="bad001")

    assert result == {
        "id": "bad001",
        "image": "images/food.jpg",
        "image_type": "invalid",
    }


def test_build_image_url_encodes_local_file(tmp_path):
    image_file = tmp_path / "tongue.jpg"
    image_file.write_bytes(b"fake-jpeg")

    url = image.build_image_url(str(image_file))

    assert url == f"data:image/jpeg;base64,{base64.b64encode(b'fake-jpeg').decode('utf-8')}"


def test_extract_json_object_rejects_missing_json():
    with pytest.raises(ValueError, match="未找到 JSON 对象"):
        image.extract_json_object("没有结构化输出")


def test_analyze_batch_skips_failed_images_and_continues(tmp_path, monkeypatch):
    first_image = tmp_path / "001.jpg"
    second_image = tmp_path / "002.jpg"
    first_image.write_bytes(b"bad")
    second_image.write_bytes(b"good")

    async def fake_analyze_image(image_path, image_id=None):
        if image_path.endswith("001.jpg"):
            raise RuntimeError("invalid image url")
        return {"id": image_id, "image": image_path, "image_type": "invalid"}

    monkeypatch.setattr(image, "analyze_image", fake_analyze_image)

    results = asyncio.run(image.analyze_batch(str(tmp_path), "*.jpg"))

    assert results == [{"id": "002", "image": str(second_image), "image_type": "invalid"}]
