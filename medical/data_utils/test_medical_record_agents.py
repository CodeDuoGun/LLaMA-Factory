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
import base64
import json
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from PIL import Image

from medical.data_utils import medical_record_agents
from medical.schema.clear_record_basemodel import (
    AgentReviewResult,
    CurrentVisitHistoryResult,
    HistoryCleaningResult,
    ImageClassification,
    ImageClassificationResult,
    InspectionFinding,
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


def test_normalize_medical_history_removes_frontend_tags() -> None:
    record = {
        "new_medical_history": (
            "<p>（22,03,28）舌淡红润有齿痕苍老裂纹苔白浊带黄根厚。胃中有嘈杂，"
            "脐上有不适。大便色暗，会返胃干呕，矢气难出，有肠鸣，有肩背疼。"
            "（22,04,09）复诊一整体好转。偶有胃烧。7分饱则胃无胀了，大便有溏色暗无，"
            "干呕很少，无肠鸣了，肩背疼好转，小便正常。舌淡润带白微红有齿痕大脾裂纹苔白微厚。"
            "（22,04,30）复诊二食后好转，大便色青次数日一次，节状。偶溏。矢气多了。"
            "偶有呃气。近2天晨起会恶心返胃。舌淡润白微红大有齿痕苔白厚。</p>"
            "<p>辅助检查：220211内镜诊断：胃底息肉，慢性非萎缩性胃炎伴胃窦糜烂</p>"
            "<p><br></p>"
        )
    }

    medical_record_agents.MedicalRecordAgents.normalize_medical_history(record)

    assert record["new_medical_history"] == (
        "（22,03,28）舌淡红润有齿痕苍老裂纹苔白浊带黄根厚。胃中有嘈杂，"
        "脐上有不适。大便色暗，会返胃干呕，矢气难出，有肠鸣，有肩背疼。"
        "（22,04,09）复诊一整体好转。偶有胃烧。7分饱则胃无胀了，大便有溏色暗无，"
        "干呕很少，无肠鸣了，肩背疼好转，小便正常。舌淡润带白微红有齿痕大脾裂纹苔白微厚。"
        "（22,04,30）复诊二食后好转，大便色青次数日一次，节状。偶溏。矢气多了。"
        "偶有呃气。近2天晨起会恶心返胃。舌淡润白微红大有齿痕苔白厚。\n"
        "辅助检查：220211内镜诊断：胃底息肉，慢性非萎缩性胃炎伴胃窦糜烂"
    )
    assert "<" not in record["new_medical_history"]
    assert ">" not in record["new_medical_history"]


def test_normalize_medical_history_decodes_entities_and_removes_empty_markup() -> None:
    record = {"new_medical_history": "<div>腹胀&nbsp;三天<br/>偶有反酸</div>"}
    medical_record_agents.MedicalRecordAgents.normalize_medical_history(record)
    assert record["new_medical_history"] == "腹胀 三天\n偶有反酸"

    empty_record = {"new_medical_history": "<p><br></p>"}
    medical_record_agents.MedicalRecordAgents.normalize_medical_history(empty_record)
    assert empty_record["new_medical_history"] == ""


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
            created_agents["image_classification_model"] = kwargs.get("image_classification_model")

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
        "temperature": medical_record_agents.DEFAULT_AGENT_TEMPERATURE,
        "max_tokens": medical_record_agents.DEFAULT_AGENT_MAX_TOKENS,
    }
    assert reviewer_model is not None
    assert created_agents["reviewer_model"] is reviewer_model
    assert created_agents["enable_review"] is True
    classifier_name, classifier_kwargs, classifier_model = next(
        item for item in built_models if item[0] == medical_record_agents.CLASSIFICATION_MODEL_NAME
    )
    assert classifier_name == "qwen3.7-flash"
    assert classifier_kwargs["max_tokens"] == 512
    assert created_agents["image_classification_model"] is classifier_model
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


def test_count_planned_records_excludes_processed_and_applies_per_doctor_limit(tmp_path) -> None:
    input_path = tmp_path / "records.json"
    input_path.write_text(
        json.dumps(
            [
                {"id": "43-1", "order_sn": "ORDER-43-1", "doctor_id": "43"},
                {"id": "43-2", "order_sn": "ORDER-43-2", "doctor_id": "43"},
                {"id": "43-3", "order_sn": "ORDER-43-3", "doctor_id": "43"},
                {"id": "52-1", "order_sn": "ORDER-52-1", "doctor_id": "52"},
            ]
        ),
        encoding="utf-8",
    )

    planned = medical_record_agents.count_planned_records(
        input_path,
        {"43"},
        2,
        set(),
        {"ORDER-43-1"},
    )

    assert planned == 2


def test_select_reprocess_records_includes_failures_and_missing_histories(tmp_path) -> None:
    doctor_dir = tmp_path / "doctor_43_朱子奇"
    doctor_dir.mkdir()
    (doctor_dir / medical_record_agents.FAILURE_FILE_NAME).write_text(
        json.dumps(
            {
                "id": "failed",
                "order_sn": "FAILED-1",
                "doctor_id": "43",
                "doctor_name": "朱子奇",
                "ai_processing_errors": [{"stage": "record", "error": "boom"}],
                "ai_processing_complete": False,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (doctor_dir / medical_record_agents.DOCTOR_RECORD_FILE_NAME).write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "missing",
                        "order_sn": "MISSING-1",
                        "doctor_id": "43",
                        "doctor_name": "朱子奇",
                        "doc_ass_stu_appeal": "",
                        "ai_patient_appeal": "",
                        "new_medical_history": "现病史",
                        "ai_new_medical_history": "AI 现病史",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "complete",
                        "order_sn": "COMPLETE-1",
                        "doctor_id": "43",
                        "doctor_name": "朱子奇",
                        "doc_ass_stu_appeal": "主诉",
                        "ai_patient_appeal": "AI 主诉",
                        "new_medical_history": "现病史",
                        "ai_new_medical_history": "AI 现病史",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    selected = medical_record_agents.select_reprocess_records(
        tmp_path,
        {"43"},
        include_failures=True,
        include_missing_histories=True,
    )

    assert [record["order_sn"] for record in selected] == ["FAILED-1", "MISSING-1"]
    assert "ai_processing_errors" not in selected[0]
    assert "ai_processing_complete" not in selected[0]


def test_replace_jsonl_records_by_order_sn_replaces_existing_and_appends_new(tmp_path) -> None:
    path = tmp_path / medical_record_agents.DOCTOR_RECORD_FILE_NAME
    path.write_text(
        "\n".join(
            [
                json.dumps({"order_sn": "A", "value": "old-a"}, ensure_ascii=False),
                json.dumps({"order_sn": "B", "value": "old-b"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    medical_record_agents._replace_jsonl_records_by_order_sn(
        path,
        [
            {"order_sn": "B", "value": "new-b"},
            {"order_sn": "C", "value": "new-c"},
        ],
    )

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {"order_sn": "A", "value": "old-a"},
        {"order_sn": "B", "value": "new-b"},
        {"order_sn": "C", "value": "new-c"},
    ]


def test_remove_jsonl_records_by_order_sn_removes_successful_failures(tmp_path) -> None:
    path = tmp_path / medical_record_agents.FAILURE_FILE_NAME
    path.write_text(
        "\n".join(
            [
                json.dumps({"order_sn": "FAILED-1"}, ensure_ascii=False),
                json.dumps({"order_sn": "FAILED-2"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    medical_record_agents._remove_jsonl_records_by_order_sn(path, {"FAILED-1"})

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records == [{"order_sn": "FAILED-2"}]


def test_progress_log_message_contains_bar_counts_speed_and_eta() -> None:
    message = medical_record_agents._progress_log_message(
        completed=25,
        total=100,
        succeeded=20,
        failed=3,
        filtered=2,
        skipped_existing=7,
        elapsed=300,
        scope="43",
    )

    assert "[PROGRESS] doctor=43 [=====...............]  25.0%" in message
    assert "completed=25/100 success=20 failed=3 filtered=2 skipped_existing=7" in message
    assert "speed=5.00 records/min" in message
    assert "elapsed=5m00s eta=15m00s" in message


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


def test_image_classification_message_deduplicates_urls_and_content(tmp_path) -> None:
    first_image = tmp_path / "first.jpg"
    duplicate_image = tmp_path / "duplicate.jpg"
    other_image = tmp_path / "other.jpg"
    Image.new("RGB", (1024, 256), "red").save(first_image, format="JPEG")
    duplicate_image.write_bytes(first_image.read_bytes())
    Image.new("RGB", (100, 100), "blue").save(other_image, format="JPEG")

    batches = asyncio.run(
        medical_record_agents._image_classification_message(
            [
                str(first_image),
                "",
                f" {first_image} ",
                str(duplicate_image),
                str(other_image),
            ],
            tmp_path,
        )
    )

    assert len(batches) == 1
    message, images = batches[0]
    assert images == [
        str(first_image),
        str(other_image),
    ]
    image_blocks = [block for block in message.content if block.get("type") == "image_url"]
    assert len(image_blocks) == 2
    assert message.content[0]["text"].startswith("下面共有 2 张图片")
    resized_data_url = image_blocks[0]["image_url"]["url"]
    resized_image = Image.open(BytesIO(base64.b64decode(resized_data_url.split(",", 1)[1])))
    assert resized_image.size == (512, 128)


def test_image_classification_message_batches_at_most_eight_images(tmp_path) -> None:
    images = []
    for index in range(9):
        path = tmp_path / f"{index}.jpg"
        color = index * 25
        Image.new("RGB", (16, 16), (color, color, color)).save(path, format="JPEG")
        images.append(str(path))

    batches = asyncio.run(medical_record_agents._image_classification_message(images, tmp_path))

    assert [len(batch_images) for _, batch_images in batches] == [8, 1]


def test_classify_images_merges_batched_results_with_original_images(tmp_path) -> None:
    images = []
    for index in range(9):
        path = tmp_path / f"classify-{index}.jpg"
        color = index * 25
        Image.new("RGB", (32, 16), (color, color, color)).save(path, format="JPEG")
        images.append(str(path))

    agents = object.__new__(medical_record_agents.MedicalRecordAgents)
    agents.input_dir = tmp_path
    agents.reviewer_agent = None
    agents.image_classifier_agent = FakeStructuredAgent(
        [
            ImageClassificationResult(
                images=[
                    ImageClassification(image_index=index, image_type="舌")
                    for index in range(1, 9)
                ]
            ),
            ImageClassificationResult(
                images=[ImageClassification(image_index=1, image_type="检验检查报告类")]
            ),
        ]
    )

    grouped = asyncio.run(agents.classify_images({"tongue_face_img": images}))

    assert grouped["舌"] == images[:8]
    assert grouped["检验检查报告类"] == images[8:]


def test_process_record_batch_isolates_errors_and_timeouts() -> None:
    class FakeAgents:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        async def process(self, record, **kwargs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                if record["id"] == "error":
                    await asyncio.sleep(0)
                    raise ValueError("broken")
                if record["id"] == "timeout":
                    await asyncio.sleep(0.1)
                else:
                    await asyncio.sleep(0.01)
                return dict(record)
            finally:
                self.active -= 1

    def batch_item(record_id):
        return {
            "record": {"id": record_id, "doctor_id": "43"},
            "record_id": record_id,
            "identity": f"43:{record_id}",
            "order_sn": "",
            "patient_key": f"43:patient-{record_id}",
            "extract_current_history": False,
            "use_rag": False,
        }

    agents = FakeAgents()
    outcomes = asyncio.run(
        medical_record_agents._process_record_batch(
            agents,
            [batch_item("ok"), batch_item("error"), batch_item("timeout")],
            {},
            record_timeout=0.03,
        )
    )

    assert agents.max_active == 3
    assert outcomes[0][2] is None
    assert isinstance(outcomes[1][2], ValueError)
    assert isinstance(outcomes[2][2], TimeoutError)
    assert outcomes[1][1]["ai_processing_errors"][0]["stage"] == "record"
    assert "病历处理超过 0.03 秒" in outcomes[2][1]["ai_processing_errors"][0]["error"]


def test_process_record_batch_keeps_same_patient_visits_sequential() -> None:
    seen_previous_diagnoses = []

    class FakeAgents:
        async def process(self, record, **kwargs):
            seen_previous_diagnoses.append(record.get("__previous_diagnoses"))
            await asyncio.sleep(0)
            enriched = dict(record)
            enriched["ai_diagnosis_illness"] = record["diagnosis_illness"]
            return enriched

    def batch_item(record_id, diagnosis):
        return {
            "record": {
                "id": record_id,
                "doctor_id": "43",
                "diagnosis_illness": diagnosis,
            },
            "record_id": record_id,
            "identity": f"43:{record_id}",
            "order_sn": "",
            "patient_key": "43:patient-1",
            "extract_current_history": False,
            "use_rag": False,
        }

    outcomes = asyncio.run(
        medical_record_agents._process_record_batch(
            FakeAgents(),
            [batch_item("visit-1", "诊断一"), batch_item("visit-2", "诊断二")],
            {},
            record_timeout=0,
        )
    )

    assert [outcome[2] for outcome in outcomes] == [None, None]
    assert seen_previous_diagnoses == [
        None,
        {
            "diagnosis_illness": "诊断一",
            "diagnosis_disease": "",
            "diagnosis_sickness": "",
        },
    ]


def test_run_stage_logs_doctor_name(monkeypatch) -> None:
    messages = []
    monkeypatch.setattr(
        medical_record_agents,
        "logger",
        SimpleNamespace(info=messages.append),
    )
    agents = object.__new__(medical_record_agents.MedicalRecordAgents)
    agents.stage_timeout = 0

    asyncio.run(agents._run_stage("image_classification", "ORDER-1", "张医生", lambda: None))

    assert messages[0] == (
        "[STAGE START] record=ORDER-1 doctor_name=张医生 stage=image_classification"
    )
    assert messages[1].startswith(
        "[STAGE END] record=ORDER-1 doctor_name=张医生 stage=image_classification elapsed="
    )


def test_process_runs_current_history_and_image_pipeline_concurrently() -> None:
    agents = object.__new__(medical_record_agents.MedicalRecordAgents)
    agents.stage_timeout = 0
    history_started = asyncio.Event()
    image_started = asyncio.Event()

    agents.normalize_diagnoses = lambda record: None

    async def classify(record):
        await asyncio.wait_for(history_started.wait(), timeout=0.2)
        image_started.set()
        return {category: [] for category in medical_record_agents.IMAGE_CATEGORIES}

    async def extract_current_history(record, enabled):
        history_started.set()
        await asyncio.wait_for(image_started.wait(), timeout=0.2)

    agents.classify_images = classify
    agents.extract_current_visit_history = extract_current_history
    agents.analyze_inspection_images = lambda record, images=None: None
    agents.analyze_tongue_face_images = lambda record, images=None: None
    agents.clean_histories = lambda record: None
    agents.complete_diagnoses = lambda record: None
    agents.extract_clinical_fields = lambda record, use_rag: None

    asyncio.run(
        asyncio.wait_for(
            agents.process(
                {
                    "order_sn": "ORDER-1",
                    "doctor_name": "张医生",
                    "new_medical_history": "复诊病史",
                },
                extract_current_history=True,
            ),
            timeout=1,
        )
    )

    assert history_started.is_set()
    assert image_started.is_set()


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


def test_clean_histories_fills_missing_complaint_and_history_from_context() -> None:
    agents = object.__new__(medical_record_agents.MedicalRecordAgents)
    agents.reviewer_agent = None
    agents.history_agent = FakeStructuredAgent(
        [
            HistoryCleaningResult(
                patient_appeal="",
                new_medical_history="",
                old_medical_history="胃炎病史多年",
            )
        ]
    )
    record = {
        "order_sn": "ORDER-1",
        "doc_ass_stu_appeal": "",
        "patient_appeal": "胃脘胀痛伴反酸1周",
        "new_medical_history": "",
        "old_medical_history": "胃炎病史多年",
        "allergic_history": "",
        "personal_history": "",
        "special_history": "",
        "family_history": "",
        "ai_inspection_report_img": {
            "reports": [
                {
                    "是否有效检查报告": True,
                    "report_name": "胃镜",
                    "报告日期": "2021-06-01",
                    "解析结果": "胃窦黏膜充血",
                    "报告结论": "慢性胃炎",
                }
            ],
            "现病史检查证据摘要": "胃镜提示慢性胃炎",
        },
        "ai_tongue_face_img": {
            "tongue": {
                "legal": "是",
                "tongue_body": {"color": "淡红"},
            }
        },
    }

    asyncio.run(agents.clean_histories(record))
    prompt_content = agents.history_agent.calls[0]["messages"][0].content

    assert "医生撰写主诉 doc_ass_stu_appeal：未记录" in prompt_content
    assert "患者主诉 patient_appeal：胃脘胀痛伴反酸1周" in prompt_content
    assert "原始本次现病史 new_medical_history：未记录" in prompt_content
    assert record["ai_patient_appeal"] == "胃脘胀痛伴反酸1周"
    assert record["ai_new_medical_history"]
    assert "胃脘胀痛伴反酸1周" in record["ai_new_medical_history"]
    assert "胃炎病史多年" in record["ai_new_medical_history"]
    assert "胃镜提示慢性胃炎" in record["ai_new_medical_history"]


def test_clean_histories_keeps_model_corrected_non_empty_complaint_and_history() -> None:
    agents = object.__new__(medical_record_agents.MedicalRecordAgents)
    agents.reviewer_agent = None
    agents.history_agent = FakeStructuredAgent(
        [
            HistoryCleaningResult(
                patient_appeal="反复胃脘胀痛1年余，加重1周",
                new_medical_history="患者反复胃脘胀痛1年余，近1周加重，伴反酸。饮食未涉及，睡眠未涉及，二便未涉及。舌象未涉及。脉象未涉及。",
            )
        ]
    )
    record = {
        "doc_ass_stu_appeal": "胃病",
        "patient_appeal": "胃痛反酸，一年多，最近一周重",
        "new_medical_history": "胃脘胀痛反复，近期加重。",
        "old_medical_history": "",
        "allergic_history": "",
        "personal_history": "",
        "special_history": "",
        "family_history": "",
        "ai_inspection_report_img": {},
        "ai_tongue_face_img": {},
    }

    asyncio.run(agents.clean_histories(record))

    assert record["ai_patient_appeal"] == "反复胃脘胀痛1年余，加重1周"
    assert record["ai_new_medical_history"].startswith("患者反复胃脘胀痛1年余")


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


def test_analyze_inspection_images_merges_structured_batches_without_ocr_summary_call(tmp_path) -> None:
    agents = object.__new__(medical_record_agents.MedicalRecordAgents)
    fake_vlm = FakeVlmModel()
    agents.input_dir = tmp_path
    agents.vlm_model = fake_vlm
    agents.reviewer_agent = None
    agents.inspection_agent = FakeStructuredAgent(
        [
            InspectionResult(
                reports=[InspectionFinding(report_name="报告一", is_valid_report=True)],
                history_evidence_summary="证据一",
            ),
            InspectionResult(
                reports=[InspectionFinding(report_name="报告二", is_valid_report=True)],
                history_evidence_summary="证据二",
            ),
        ]
    )

    record = {"see_doc_time": "2026-07-30"}
    images = [
        f"https://example.com/report-{index}.jpg"
        for index in range(medical_record_agents.MAX_VLM_IMAGES_PER_REQUEST + 1)
    ]

    asyncio.run(agents.analyze_inspection_images(record, images=images))

    assert fake_vlm.image_counts == []
    assert len(agents.inspection_agent.calls) == 2
    batch_image_counts = [
        sum(1 for block in call["messages"][0].content if block.get("type") == "image_url")
        for call in agents.inspection_agent.calls
    ]
    assert batch_image_counts == [medical_record_agents.MAX_VLM_IMAGES_PER_REQUEST, 1]
    assert [report["report_name"] for report in record["ai_inspection_report_img"]["reports"]] == [
        "报告一",
        "报告二",
    ]
    assert record["ai_inspection_report_img"]["现病史检查证据摘要"] == "证据一\n证据二"


def test_analyze_tongue_face_images_ocr_batches_when_over_request_limit(tmp_path) -> None:
    agents = object.__new__(medical_record_agents.MedicalRecordAgents)
    fake_vlm = FakeVlmModel()
    agents.input_dir = tmp_path
    agents.vlm_model = fake_vlm
    agents.reviewer_agent = None
    agents.tongue_face_agent = FakeStructuredAgent([TongueFaceResult()])

    record = {"order_sn": "ORDER-1"}
    images = [
        f"https://example.com/tongue-face-{index}.jpg"
        for index in range(medical_record_agents.MAX_VLM_IMAGES_PER_REQUEST + 1)
    ]

    asyncio.run(agents.analyze_tongue_face_images(record, images=images))

    assert fake_vlm.image_counts == [medical_record_agents.MAX_VLM_IMAGES_PER_REQUEST, 1]
    final_message = agents.tongue_face_agent.calls[0]["messages"][0]
    assert isinstance(final_message.content, str)
    assert (
        f"OCR batch with {medical_record_agents.MAX_VLM_IMAGES_PER_REQUEST} images"
        in final_message.content
    )
    assert "OCR batch with 1 images" in final_message.content
    assert "ai_tongue_face_img" in record


def test_ocr_image_batches_run_concurrently_and_keep_batch_order(tmp_path) -> None:
    agents = object.__new__(medical_record_agents.MedicalRecordAgents)
    agents.input_dir = tmp_path
    agents.reviewer_agent = None
    agents.ocr_batch_semaphore = asyncio.Semaphore(2)
    active_calls = 0
    max_active_calls = 0

    class ConcurrentVlm:
        async def ainvoke(self, messages):
            nonlocal active_calls, max_active_calls
            prompt = messages[0].content[0]["text"]
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            await asyncio.sleep(0.01)
            active_calls -= 1
            return SimpleNamespace(content=prompt.split("原始图片序号 ", 1)[1].split(" 到", 1)[0])

    agents.vlm_model = ConcurrentVlm()
    image_count = medical_record_agents.MAX_VLM_IMAGES_PER_REQUEST * 3
    images = [f"https://example.com/image-{index}.jpg" for index in range(image_count)]

    result = asyncio.run(agents._ocr_image_batches(images, "OCR", agent_name="benchmark"))

    assert max_active_calls == 2
    assert result.index("OCR 批次 1") < result.index("OCR 批次 2") < result.index("OCR 批次 3")


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
