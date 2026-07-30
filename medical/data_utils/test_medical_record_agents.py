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

import argparse
import asyncio
import json
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from medical.data_utils import medical_record_agents
from medical.schema.clear_record_basemodel import (
    AgentReviewResult,
    CurrentVisitHistoryResult,
    ImageClassification,
    ImageClassificationResult,
    InspectionResult,
    TongueFaceResult,
)


class FakeStructuredAgent:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def ainvoke(self, payload):
        self.calls.append(payload)
        response = self.responses.pop(0)
        return {
            "structured_response": response,
            "messages": [SimpleNamespace(content=f"returned {type(response).__name__}", tool_calls=[])],
        }


def test_structured_agent_review_revises_failed_result() -> None:
    source_agent = FakeStructuredAgent(
        [
            CurrentVisitHistoryResult(new_medical_history="混入上次就诊内容"),
            CurrentVisitHistoryResult(new_medical_history="本次咳嗽较前减轻"),
        ]
    )
    reviewer_agent = FakeStructuredAgent(
        [
            AgentReviewResult(
                followed_prompt=False,
                response_meets_requirements=False,
                passed=False,
                issues=["混入上次就诊内容"],
                revision_instructions="删除上次就诊内容，仅保留本次咳嗽变化。",
            ),
            AgentReviewResult(followed_prompt=True, response_meets_requirements=True, passed=True),
        ]
    )

    result = asyncio.run(
        medical_record_agents._invoke_structured_agent(
            source_agent,
            HumanMessage(content="本次：咳嗽较前减轻；上次：发热。"),
            CurrentVisitHistoryResult,
            reviewer_agent=reviewer_agent,
            agent_name="current_visit_history_agent",
            system_prompt="仅提取本次就诊内容。",
        )
    )

    assert result.new_medical_history == "本次咳嗽较前减轻"
    assert len(source_agent.calls) == 2
    revision_content = source_agent.calls[1]["messages"][1].content
    assert "删除上次就诊内容" in revision_content
    review_content = reviewer_agent.calls[0]["messages"][0].content
    assert "【可观察执行轨迹】" in review_content
    assert "returned CurrentVisitHistoryResult" in review_content


def test_json_object_agent_review_revises_failed_result() -> None:
    source_model = SimpleNamespace()
    source_model.calls = []
    source_responses = iter(
        [
            '{"new_medical_history":"混入既往内容"}',
            '{"new_medical_history":"本次腹胀"}',
        ]
    )

    async def ainvoke(messages):
        source_model.calls.append(messages)
        return SimpleNamespace(content=next(source_responses))

    source_model.ainvoke = ainvoke
    reviewer_agent = FakeStructuredAgent(
        [
            AgentReviewResult(
                followed_prompt=False,
                response_meets_requirements=False,
                passed=False,
                issues=["包含既往内容"],
                revision_instructions="仅保留本次腹胀。",
            ),
            AgentReviewResult(followed_prompt=True, response_meets_requirements=True, passed=True),
        ]
    )

    result = asyncio.run(
        medical_record_agents._invoke_json_object_model(
            source_model,
            "仅提取本次内容。",
            HumanMessage(content="本次腹胀；既往头痛。"),
            CurrentVisitHistoryResult,
            reviewer_agent=reviewer_agent,
            agent_name="json_agent",
        )
    )

    assert result.new_medical_history == "本次腹胀"
    assert len(source_model.calls) == 2


def test_process_records_skips_existing_order_sn_and_logs_it(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "records.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "id": "new-id",
                    "order_sn": "ORDER-43",
                    "doctor_id": "43",
                    "doctor_name": "朱子奇",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "processed_data"
    result_dir = output_dir / "doctor_43_朱子奇"
    result_dir.mkdir(parents=True)
    (result_dir / medical_record_agents.DOCTOR_RECORD_FILE_NAME).write_text(
        json.dumps(
            {
                "id": "old-id",
                "order_sn": "ORDER-43",
                "doctor_id": "43",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    built_models = []

    def fake_build_model(model_name, args, **kwargs):
        model = object()
        built_models.append((model_name, kwargs, model))
        return model

    monkeypatch.setattr(medical_record_agents, "build_model", fake_build_model)
    created_agents = {}

    class FakeAgents:
        def __init__(self, *args, **kwargs):
            created_agents["reviewer_model"] = kwargs.get("reviewer_model")
            created_agents["enable_review"] = kwargs.get("enable_review")

        async def process(self, *args, **kwargs):
            raise AssertionError("已处理的 order_sn 不应再次进入处理流程")

    messages = []

    class FakeLogger:
        @staticmethod
        def info(message):
            messages.append(message)

        @staticmethod
        def error(message):
            raise AssertionError(message)

    monkeypatch.setattr(medical_record_agents, "MedicalRecordAgents", FakeAgents)
    monkeypatch.setattr(medical_record_agents, "logger", FakeLogger())
    args = argparse.Namespace(
        output_dir=output_dir,
        doctor_id=[],
        model="text-model",
        diagnosis_model="diagnosis-model",
        diagnosis_base_url="",
        diagnosis_api_key="",
        vlm_model="vlm-model",
        vlm_base_url="",
        base_url="",
        vlm_api_key="",
        reviewer_model="kimi/kimi-k3",
        reviewer_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        reviewer_api_key="test-dashscope-key",
        enable_review=True,
        api_key="",
        input=input_path,
        reprocess=False,
        limit=0,
        use_rag=False,
        interval=0,
        fail_fast=False,
    )

    assert asyncio.run(medical_record_agents.process_records(args)) == (0, 0, 1)
    reviewer_name, reviewer_kwargs, reviewer_model = built_models[-1]
    assert reviewer_name == "kimi/kimi-k3"
    assert reviewer_kwargs == {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "test-dashscope-key",
    }
    assert reviewer_model is not None
    assert created_agents["reviewer_model"] is reviewer_model
    assert created_agents["enable_review"] is True
    assert messages == ["数据已经处理过，【SKIP】 ORDER-43"]


def test_process_records_limits_attempts_per_doctor(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "records.json"
    input_path.write_text(
        json.dumps(
            [
                {"id": "43-1", "order_sn": "ORDER-43-1", "doctor_id": "43", "doctor_name": "甲"},
                {"id": "43-2", "order_sn": "ORDER-43-2", "doctor_id": "43", "doctor_name": "甲"},
                {"id": "52-1", "order_sn": "ORDER-52-1", "doctor_id": "52", "doctor_name": "乙"},
                {"id": "52-2", "order_sn": "ORDER-52-2", "doctor_id": "52", "doctor_name": "乙"},
                {"id": "99-1", "order_sn": "ORDER-99-1", "doctor_id": "99", "doctor_name": "丙"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    processed_order_sns = []

    monkeypatch.setattr(medical_record_agents, "build_model", lambda *args, **kwargs: object())

    class FakeAgents:
        def __init__(self, *args, **kwargs):
            pass

        async def process(self, record, **kwargs):
            processed_order_sns.append(record["order_sn"])
            return dict(record)

    monkeypatch.setattr(medical_record_agents, "MedicalRecordAgents", FakeAgents)
    args = argparse.Namespace(
        output_dir=tmp_path / "processed_data",
        doctor_id=["43", "52"],
        model="text-model",
        diagnosis_model="diagnosis-model",
        diagnosis_base_url="",
        diagnosis_api_key="",
        vlm_model="vlm-model",
        vlm_base_url="",
        base_url="",
        vlm_api_key="",
        reviewer_model="kimi/kimi-k3",
        reviewer_base_url="",
        reviewer_api_key="",
        enable_review=False,
        api_key="",
        input=input_path,
        reprocess=False,
        limit=1,
        use_rag=False,
        interval=0,
        fail_fast=False,
        timeout=120,
        max_retries=2,
    )

    assert asyncio.run(medical_record_agents.process_records(args)) == (2, 0, 0)
    assert processed_order_sns == ["ORDER-43-1", "ORDER-52-1"]


def test_review_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["medical_record_agents.py"])

    args = medical_record_agents.parse_args()

    assert args.enable_review is False


def test_should_filter_record_requires_all_key_context_to_be_empty() -> None:
    empty_record = {
        "doc_ass_stu_appeal": " ",
        "patient_appeal": None,
        "new_medical_history": "",
        "diagnosis_illness": "",
    }

    assert medical_record_agents._should_filter_record(empty_record) is True
    assert medical_record_agents._should_filter_record({**empty_record, "doc_ass_stu_appeal": "反复咳嗽"}) is True
    assert medical_record_agents._should_filter_record({**empty_record, "patient_appeal": "胃胀"}) is False
    assert medical_record_agents._should_filter_record({**empty_record, "new_medical_history": "咳嗽三天"}) is False
    assert medical_record_agents._should_filter_record({**empty_record, "diagnosis_illness": "肺炎"}) is False


def test_should_filter_record_filters_short_patient_appeal_without_history_or_western_diagnosis() -> None:
    empty_record = {
        "patient_appeal": "反复咳嗽伴咽痛三天",
        "new_medical_history": "",
        "diagnosis_illness": "",
    }

    assert medical_record_agents._should_filter_record(empty_record) is True
    assert (
        medical_record_agents._should_filter_record(
            {**empty_record, "patient_appeal": "反复咳嗽伴咽痛三天，夜间加重，伴少量白痰，无发热"}
        )
        is False
    )
    assert medical_record_agents._should_filter_record({**empty_record, "new_medical_history": "咳嗽三天"}) is False
    assert medical_record_agents._should_filter_record({**empty_record, "diagnosis_illness": "上呼吸道感染"}) is False


def test_process_writes_empty_record_to_filter_json_without_running_agents(tmp_path) -> None:
    agents = object.__new__(medical_record_agents.MedicalRecordAgents)
    agents.filter_path = tmp_path / "filter.json"
    agents.stage_timeout = 0
    record = {
        "order_sn": "EMPTY-1",
        "doctor_id": "43",
        "patient_appeal": "",
        "new_medical_history": None,
        "diagnosis_illness": " ",
    }

    first = asyncio.run(agents.process(record))
    second = asyncio.run(agents.process(record))

    assert first["ai_processing_filtered"] is True
    assert second["ai_processing_filtered"] is True
    assert json.loads(agents.filter_path.read_text(encoding="utf-8")) == [record]
    _, processed_order_sns = medical_record_agents.load_processed_record_keys(tmp_path)
    assert processed_order_sns == {"EMPTY-1"}


def test_record_images_collects_all_four_fields_and_deduplicates() -> None:
    record = {
        "tongue_face_img": [{"img": "https://example.com/tongue.jpg"}],
        "admin_face_img": ["https://example.com/face.jpg"],
        "admin_report_img": [{"url": "https://example.com/report.jpg"}],
        "inspection_report_img": [
            "https://example.com/report.jpg",
            {"image_url": "https://example.com/other.jpg"},
        ],
    }

    assert medical_record_agents._record_images(record) == [
        "https://example.com/tongue.jpg",
        "https://example.com/face.jpg",
        "https://example.com/report.jpg",
        "https://example.com/other.jpg",
    ]


@pytest.mark.parametrize(
    ("report_date", "visit_time", "expected"),
    [
        ("2021-03-10", "2021-07-06 22:34:34", "3个月前"),
        ("2021-07-01", "2021-07-06 22:34:34", "5天前"),
        ("2019-07-06", "2021-07-06 22:34:34", "2年前"),
        ("2021-07-06", "2021-07-06 22:34:34", "当天"),
        ("2021-07-08", "2021-07-06 22:34:34", "2天后"),
        ("未识别", "2021-07-06 22:34:34", ""),
    ],
)
def test_relative_report_time(report_date, visit_time, expected) -> None:
    assert medical_record_agents._relative_report_time(report_date, visit_time) == expected


def test_visit_time_prefers_see_doc_time() -> None:
    assert (
        medical_record_agents._visit_time(
            {
                "see_doc_time": "2021-07-06 22:34:34",
                "start_time": "2021-09-25 00:00:00",
                "created_at": "2021-07-06 20:07:02",
            }
        )
        == "2021-07-06 22:34:34"
    )


def test_format_inspection_result_text_keeps_valid_reports_only() -> None:
    result = {
        "reports": [
            {
                "图片类别": "影像检查报告",
                "是否有效检查报告": True,
                "report_name": "胸部CT",
                "报告日期": "2021-03-10",
                "相对就诊时间": "3个月前",
                "解析结果": "右肺见结节影",
                "异常指标": [],
                "报告结论": "右肺结节",
            },
            {
                "图片类别": "非检查报告",
                "是否有效检查报告": False,
                "report_name": "处方",
                "解析结果": "不应进入上下文",
            },
        ],
        "现病史检查证据摘要": "胸部CT提示右肺结节",
    }

    assert medical_record_agents.format_inspection_result_text(result) == (
        "有效检查报告1：图片类别：影像检查报告；报告名称：胸部CT；报告日期：2021-03-10；"
        "相对就诊时间：3个月前；解析结果：右肺见结节影；报告结论：右肺结节。\n"
        "现病史检查证据摘要：胸部CT提示右肺结节。"
    )


def test_format_tongue_face_result_text_uses_natural_language_sections() -> None:
    result = {
        "tongue": {
            "legal": "是",
            "tongue_body": {"color": "淡红", "shape": ["胖", "齿痕"]},
        },
        "face": {},
        "lesions": {"location": ["面颊"], "color": ["红"]},
    }

    assert medical_record_agents.format_tongue_face_result_text(result) == (
        "舌象：舌质颜色：淡红；舌体形态：胖、齿痕。\n患处：患处部位：面颊；患处颜色：红。"
    )


def test_group_classified_images_routes_categories_and_keeps_other_separate() -> None:
    images = ["tongue.jpg", "face.jpg", "lesion.jpg", "report.jpg", "landscape.jpg"]
    result = ImageClassificationResult(
        images=[
            ImageClassification(image_index=1, image_type="舌"),
            ImageClassification(image_index=2, image_type="面"),
            ImageClassification(image_index=3, image_type="患处"),
            ImageClassification(image_index=4, image_type="检验检查报告类"),
            ImageClassification(image_index=5, image_type="其他类"),
        ]
    )

    grouped = medical_record_agents._group_classified_images(result, images)

    assert grouped == {
        "舌": ["tongue.jpg"],
        "面": ["face.jpg"],
        "患处": ["lesion.jpg"],
        "检验检查报告类": ["report.jpg"],
        "其他类": ["landscape.jpg"],
    }


def test_group_classified_images_rejects_missing_result() -> None:
    result = ImageClassificationResult(images=[ImageClassification(image_index=1, image_type="舌")])

    with pytest.raises(ValueError, match="缺少序号"):
        medical_record_agents._group_classified_images(result, ["tongue.jpg", "face.jpg"])


class FakeVlmModel:
    def __init__(self):
        self.image_counts = []

    async def ainvoke(self, messages):
        content = messages[0].content
        self.image_counts.append(sum(1 for block in content if block.get("type") == "image_url"))
        return SimpleNamespace(content=f"OCR batch with {self.image_counts[-1]} images")


def test_analyze_inspection_images_ocr_batches_when_more_than_five_images(tmp_path) -> None:
    agents = object.__new__(medical_record_agents.MedicalRecordAgents)
    fake_vlm = FakeVlmModel()
    agents.input_dir = tmp_path
    agents.vlm_model = fake_vlm
    agents.reviewer_agent = None
    agents.inspection_agent = FakeStructuredAgent([InspectionResult(history_evidence_summary="OCR 汇总")])

    record = {"see_doc_time": "2026-07-30"}
    images = [f"https://example.com/report-{index}.jpg" for index in range(6)]

    asyncio.run(agents.analyze_inspection_images(record, images=images))

    assert fake_vlm.image_counts == [5, 1]
    final_message = agents.inspection_agent.calls[0]["messages"][0]
    assert isinstance(final_message.content, str)
    assert "OCR batch with 5 images" in final_message.content
    assert "OCR batch with 1 images" in final_message.content
    assert record["ai_inspection_report_img"]["现病史检查证据摘要"] == "OCR 汇总"


def test_analyze_tongue_face_images_ocr_batches_when_more_than_five_images(tmp_path) -> None:
    agents = object.__new__(medical_record_agents.MedicalRecordAgents)
    fake_vlm = FakeVlmModel()
    agents.input_dir = tmp_path
    agents.vlm_model = fake_vlm
    agents.reviewer_agent = None
    agents.tongue_face_agent = FakeStructuredAgent([TongueFaceResult()])

    record = {"order_sn": "ORDER-1"}
    images = [f"https://example.com/tongue-face-{index}.jpg" for index in range(6)]

    asyncio.run(agents.analyze_tongue_face_images(record, images=images))

    assert fake_vlm.image_counts == [5, 1]
    final_message = agents.tongue_face_agent.calls[0]["messages"][0]
    assert isinstance(final_message.content, str)
    assert "OCR batch with 5 images" in final_message.content
    assert "OCR batch with 1 images" in final_message.content
    assert "ai_tongue_face_img" in record


def test_process_classifies_once_and_filters_other_images() -> None:
    agents = object.__new__(medical_record_agents.MedicalRecordAgents)
    agents.stage_timeout = 0
    calls = {}
    classification_calls = []

    agents.normalize_diagnoses = lambda record: None

    async def classify(record):
        classification_calls.append(record)
        return {
            "舌": ["tongue.jpg"],
            "面": ["face.jpg"],
            "患处": ["lesion.jpg"],
            "检验检查报告类": ["report.jpg"],
            "其他类": ["landscape.jpg"],
        }

    agents.classify_images = classify

    def analyze_inspection(record, images=None):
        calls["inspection"] = images
        record["ai_inspection_report_img"] = {}

    def analyze_tongue_face(record, images=None):
        calls["tongue_face"] = images
        record["ai_tongue_face_img"] = {}

    agents.analyze_inspection_images = analyze_inspection
    agents.analyze_tongue_face_images = analyze_tongue_face
    agents.extract_current_visit_history = lambda record, enabled: None
    agents.clean_histories = lambda record: None
    agents.complete_diagnoses = lambda record: None
    agents.extract_clinical_fields = lambda record, use_rag: None

    asyncio.run(agents.process({}, extract_current_history=False))

    assert calls == {
        "inspection": ["report.jpg"],
        "tongue_face": ["tongue.jpg", "face.jpg", "lesion.jpg"],
    }
    assert len(classification_calls) == 1
    assert "landscape.jpg" not in calls["inspection"] + calls["tongue_face"]
