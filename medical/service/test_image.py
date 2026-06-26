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


def test_build_image_url_encodes_local_file(tmp_path):
    image_file = tmp_path / "tongue.jpg"
    image_file.write_bytes(b"fake-jpeg")

    url = image.build_image_url(str(image_file))

    assert url == f"data:image/jpeg;base64,{base64.b64encode(b'fake-jpeg').decode('utf-8')}"


def test_extract_json_object_rejects_missing_json():
    with pytest.raises(ValueError, match="未找到 JSON 对象"):
        image.extract_json_object("没有结构化输出")
