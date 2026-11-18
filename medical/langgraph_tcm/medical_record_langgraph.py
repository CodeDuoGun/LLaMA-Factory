#!/usr/bin/env python3
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

"""LangGraph workflow for the medical record AI processing pipeline."""

from __future__ import annotations

import operator
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medical.data_utils import medical_record_agents as record_agents  # noqa: E402


class MedicalRecordGraphState(TypedDict, total=False):
    """State passed between the per-record workflow nodes."""

    record: dict[str, Any]
    stage_names: list[str] | tuple[str, ...] | None
    selected_stages: tuple[str, ...]
    use_rag: bool
    skip_record_filter: bool
    record_id: str
    doctor_name: str
    original_history_present: bool
    original_history: Any
    filtered: bool
    classified_images: dict[str, list[str]]
    inspection_result: Any
    tongue_face_result: Any
    errors: Annotated[list[dict[str, str]], operator.add]


class MedicalRecordGraphInput(TypedDict):
    """Public input accepted by the per-record graph."""

    record: dict[str, Any]
    stage_names: NotRequired[list[str] | tuple[str, ...] | None]
    use_rag: NotRequired[bool]
    skip_record_filter: NotRequired[bool]


class MedicalRecordGraphOutput(TypedDict):
    """Public output returned after the internal workflow state is finalized."""

    record: dict[str, Any]


class MedicalRecordJsonlGraphState(TypedDict, total=False):
    """State used by the file-level workflow."""

    input_file: str
    records: list[dict[str, Any]]
    processed_records: list[dict[str, Any]]
    record_count: int
    stage_names: list[str] | tuple[str, ...] | None
    use_rag: bool
    skip_record_filter: bool


class MedicalRecordJsonlGraphInput(TypedDict):
    """Public input accepted by the JSONL workflow."""

    input_file: str
    stage_names: NotRequired[list[str] | tuple[str, ...] | None]
    use_rag: NotRequired[bool]
    skip_record_filter: NotRequired[bool]


class MedicalRecordJsonlGraphOutput(TypedDict):
    """Processed records returned by the JSONL workflow."""

    processed_records: list[dict[str, Any]]
    record_count: int


RecordOperation = Callable[[dict[str, Any], MedicalRecordGraphState], Any]


def _empty_classifications() -> dict[str, list[str]]:
    return {category: [] for category in record_agents.IMAGE_CATEGORIES}


def _stage_error(stage_name: str, error: Exception) -> dict[str, str]:
    return {"stage": stage_name, "error": f"{type(error).__name__}: {error}"}


def build_medical_record_graph(
    agents: record_agents.MedicalRecordAgents,
) -> Any:
    """Build the stage graph around an initialized MedicalRecordAgents instance.

    The classifier and the two visual analysis nodes are routed as a fan-out/fan-in
    group. All later stages run in the same order as MedicalRecordAgents.process().
    """

    def first_selected_after_images(state: MedicalRecordGraphState) -> str:
        selected = set(state["selected_stages"])
        for stage_name in record_agents.STAGE_NAMES[3:]:
            if stage_name in selected:
                return stage_name
        return "finalize"

    def next_selected_stage(state: MedicalRecordGraphState, current: str) -> str:
        selected = set(state["selected_stages"])
        current_index = record_agents.STAGE_NAMES.index(current)
        for stage_name in record_agents.STAGE_NAMES[current_index + 1 :]:
            if stage_name in selected:
                return stage_name
        return "finalize"

    def make_stage_router(current: str) -> Callable[[MedicalRecordGraphState], str]:
        def route(state: MedicalRecordGraphState) -> str:
            return next_selected_stage(state, current)

        return route

    def stage_route_map(current: str) -> dict[str, str]:
        current_index = record_agents.STAGE_NAMES.index(current)
        return {
            **{name: name for name in record_agents.STAGE_NAMES[current_index + 1 :]},
            "finalize": "finalize",
        }

    def prepare_node(state: MedicalRecordGraphState) -> dict[str, Any]:
        record = dict(state["record"])
        original_history_present = "new_medical_history" in record
        original_history = record.get("new_medical_history")
        agents.normalize_medical_history(record)
        record["__original_diagnoses"] = record_agents._diagnosis_payload(record)
        record_id = (
            record_agents._text(record.get("order_sn"))
            or record_agents._text(record.get("id"))
            or record_agents._text(record.get("record_id"))
            or "<unknown>"
        )
        doctor_name = record_agents._text(record.get("doctor_name")) or "<unknown>"
        selected_stages = record_agents.resolve_stage_names(state.get("stage_names"))
        return {
            "record": record,
            "selected_stages": selected_stages,
            "record_id": record_id,
            "doctor_name": doctor_name,
            "original_history_present": original_history_present,
            "original_history": original_history,
            "filtered": False,
            "classified_images": _empty_classifications(),
            "errors": [],
        }

    def record_filter_node(state: MedicalRecordGraphState) -> dict[str, Any]:
        record = dict(state["record"])
        if state.get("skip_record_filter", False) or not record_agents._should_filter_record(record):
            return {"record": record, "filtered": False}

        agents.filter_path.parent.mkdir(parents=True, exist_ok=True)
        record_agents.write2filter_json(record, agents.filter_path)
        record["ai_processing_filtered"] = True
        record["ai_processing_complete"] = False
        record["ai_filter_reason"] = "patient_appeal、new_medical_history、diagnosis_illness 均为空"
        return {"record": record, "filtered": True}

    async def image_classification_node(state: MedicalRecordGraphState) -> dict[str, Any]:
        classified_images = _empty_classifications()

        async def classify() -> None:
            nonlocal classified_images
            classified_images = await agents.classify_images(state["record"])

        try:
            await agents._run_stage(
                "image_classification",
                state["record_id"],
                state["doctor_name"],
                classify,
            )
        except Exception as exc:
            return {
                "classified_images": _empty_classifications(),
                "errors": [_stage_error("image_classification", exc)],
            }
        return {"classified_images": classified_images}

    async def inspection_vlm_node(state: MedicalRecordGraphState) -> dict[str, Any]:
        if "inspection_vlm" not in set(state["selected_stages"]):
            return {}
        record = dict(state["record"])
        images = state["classified_images"].get("检验检查报告类", [])
        try:
            await agents._run_stage(
                "inspection_vlm",
                state["record_id"],
                state["doctor_name"],
                lambda: agents.analyze_inspection_images(record, images),
            )
        except Exception as exc:
            return {
                "inspection_result": record.get("ai_inspection_report_img"),
                "errors": [_stage_error("inspection_vlm", exc)],
            }
        return {"inspection_result": record.get("ai_inspection_report_img")}

    async def tongue_face_vlm_node(state: MedicalRecordGraphState) -> dict[str, Any]:
        if "tongue_face_vlm" not in set(state["selected_stages"]):
            return {}
        record = dict(state["record"])
        images = [
            image
            for category in record_agents.TONGUE_FACE_IMAGE_CATEGORIES
            for image in state["classified_images"].get(category, [])
        ]
        try:
            await agents._run_stage(
                "tongue_face_vlm",
                state["record_id"],
                state["doctor_name"],
                lambda: agents.analyze_tongue_face_images(record, images),
            )
        except Exception as exc:
            return {
                "tongue_face_result": record.get("ai_tongue_face_img"),
                "errors": [_stage_error("tongue_face_vlm", exc)],
            }
        return {"tongue_face_result": record.get("ai_tongue_face_img")}

    def merge_visual_results_node(state: MedicalRecordGraphState) -> dict[str, Any]:
        record = dict(state["record"])
        selected = set(state["selected_stages"])
        if "inspection_vlm" in selected and "inspection_result" in state:
            record["ai_inspection_report_img"] = state["inspection_result"]
        if "tongue_face_vlm" in selected and "tongue_face_result" in state:
            record["ai_tongue_face_img"] = state["tongue_face_result"]
        return {"record": record}

    def make_record_stage_node(stage_name: str, operation: RecordOperation) -> Callable[
        [MedicalRecordGraphState], Any
    ]:
        async def run_stage(state: MedicalRecordGraphState) -> dict[str, Any]:
            record = dict(state["record"])
            try:
                await agents._run_stage(
                    stage_name,
                    state["record_id"],
                    state["doctor_name"],
                    lambda: operation(record, state),
                )
            except Exception as exc:
                return {"record": record, "errors": [_stage_error(stage_name, exc)]}
            return {"record": record}

        return run_stage

    def clean_histories(record: dict[str, Any], _: MedicalRecordGraphState) -> Any:
        return agents.clean_histories(record)

    def complete_diagnoses(record: dict[str, Any], _: MedicalRecordGraphState) -> Any:
        return agents.complete_diagnoses(record)

    def extract_clinical_fields(record: dict[str, Any], state: MedicalRecordGraphState) -> Any:
        return agents.extract_clinical_fields(record, state.get("use_rag", False))

    def normalize_diagnoses(record: dict[str, Any], _: MedicalRecordGraphState) -> None:
        agents.normalize_diagnoses(record)

    def finalize_node(state: MedicalRecordGraphState) -> dict[str, Any]:
        record = dict(state["record"])
        errors = state.get("errors", [])
        if errors:
            record["ai_processing_errors"] = errors
        else:
            record.pop("ai_processing_errors", None)
        record.pop("__original_diagnoses", None)
        record.pop("__previous_diagnoses", None)
        record.pop("__previous_record_context", None)
        if state.get("original_history_present", False):
            record["new_medical_history"] = state.get("original_history")
        else:
            record.pop("new_medical_history", None)
        return {"record": record}

    def route_after_filter(state: MedicalRecordGraphState) -> str:
        if state.get("filtered", False):
            return "finalize"
        selected = set(state["selected_stages"])
        if "image_classification" in selected:
            return "image_classification"
        return first_selected_after_images(state)

    def route_after_classification(_: MedicalRecordGraphState) -> list[str]:
        # Both visual nodes run as no-ops when not selected. A static join edge
        # then safely waits for every requested branch before merging results.
        return ["inspection_vlm", "tongue_face_vlm"]

    builder = StateGraph(
        MedicalRecordGraphState,
        input_schema=MedicalRecordGraphInput,
        output_schema=MedicalRecordGraphOutput,
    )
    builder.add_node("prepare", prepare_node)
    builder.add_node("record_filter", record_filter_node)
    builder.add_node("image_classification", image_classification_node)
    builder.add_node("inspection_vlm", inspection_vlm_node)
    builder.add_node("tongue_face_vlm", tongue_face_vlm_node)
    builder.add_node("merge_visual_results", merge_visual_results_node)
    builder.add_node("clinical_cleaning", make_record_stage_node("clinical_cleaning", clean_histories))
    builder.add_node("diagnosis_completion", make_record_stage_node("diagnosis_completion", complete_diagnoses))
    builder.add_node("clinical_extraction", make_record_stage_node("clinical_extraction", extract_clinical_fields))
    builder.add_node("diagnosis_normalization", make_record_stage_node("diagnosis_normalization", normalize_diagnoses))
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "record_filter")
    record_filter_targets = {
        "finalize": "finalize",
        "image_classification": "image_classification",
        **{name: name for name in record_agents.STAGE_NAMES[3:]},
    }
    visual_branch_targets = {name: name for name in ("inspection_vlm", "tongue_face_vlm")}
    builder.add_conditional_edges("record_filter", route_after_filter, record_filter_targets)
    builder.add_conditional_edges("image_classification", route_after_classification, visual_branch_targets)
    builder.add_edge(["inspection_vlm", "tongue_face_vlm"], "merge_visual_results")
    builder.add_conditional_edges(
        "merge_visual_results",
        make_stage_router("tongue_face_vlm"),
        stage_route_map("tongue_face_vlm"),
    )
    for stage_name in record_agents.STAGE_NAMES[3:]:
        if stage_name == "diagnosis_normalization":
            builder.add_edge(stage_name, "finalize")
        else:
            builder.add_conditional_edges(
                stage_name,
                make_stage_router(stage_name),
                stage_route_map(stage_name),
            )
    builder.add_edge("finalize", END)
    return builder.compile()


def build_medical_record_jsonl_graph(agents: LangGraphMedicalRecordAgents) -> Any:
    """Build a file-level graph that reads and processes every JSONL record."""

    def read_jsonl_node(state: MedicalRecordJsonlGraphState) -> dict[str, Any]:
        input_file = Path(state["input_file"]).expanduser().resolve()
        if input_file.suffix.lower() != ".jsonl":
            raise ValueError(f"输入文件必须是 .jsonl 文件: {input_file}")
        if not input_file.is_file():
            raise FileNotFoundError(f"JSONL 输入文件不存在: {input_file}")

        records = list(record_agents._iter_record_file(input_file))
        return {"records": records}

    async def process_records_node(
        state: MedicalRecordJsonlGraphState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        processed_records: list[dict[str, Any]] = []
        previous_diagnoses_by_patient: dict[str, dict[str, Any]] = {}
        records = state.get("records", [])

        for index, source_record in enumerate(records, 1):
            working_record = dict(source_record)
            patient_key = record_agents._patient_key(working_record)
            if patient_key and patient_key in previous_diagnoses_by_patient:
                previous_state = previous_diagnoses_by_patient[patient_key]
                working_record["__previous_diagnoses"] = {
                    field: record_agents._text(previous_state.get(field))
                    for field in record_agents.DIAGNOSIS_FIELDS
                }
                previous_context = previous_state.get("__record_context")
                if isinstance(previous_context, dict):
                    working_record["__previous_record_context"] = dict(previous_context)

            try:
                result = await agents.workflow.ainvoke(
                    {
                        "record": working_record,
                        "stage_names": state.get("stage_names"),
                        "use_rag": state.get("use_rag", False),
                        "skip_record_filter": state.get("skip_record_filter", False),
                    },
                    config=config,
                )
                enriched = result["record"]
            except Exception as exc:
                enriched = dict(working_record)
                enriched.pop("__previous_diagnoses", None)
                enriched.pop("__previous_record_context", None)
                enriched.pop("__original_diagnoses", None)
                enriched["ai_processing_errors"] = [
                    {"stage": "record", "error": f"{type(exc).__name__}: {exc}"}
                ]
                enriched["ai_processing_complete"] = False
                enriched["ai_record_identity"] = record_agents._record_identity(working_record, index)

            processed_records.append(enriched)
            if patient_key and enriched.get("ai_processing_filtered") is not True:
                previous_diagnoses_by_patient[patient_key] = {
                    **record_agents._diagnosis_payload(enriched, prefer_ai=True),
                    "__record_context": record_agents._record_history_context(enriched),
                }

        return {"processed_records": processed_records, "record_count": len(processed_records)}

    builder = StateGraph(
        MedicalRecordJsonlGraphState,
        input_schema=MedicalRecordJsonlGraphInput,
        output_schema=MedicalRecordJsonlGraphOutput,
    )
    builder.add_node("read_jsonl", read_jsonl_node)
    builder.add_node("process_records", process_records_node)
    builder.add_edge(START, "read_jsonl")
    builder.add_edge("read_jsonl", "process_records")
    builder.add_edge("process_records", END)
    return builder.compile()


class LangGraphMedicalRecordAgents(record_agents.MedicalRecordAgents):
    """Drop-in record processor that executes each stage as a LangGraph node."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.workflow = build_medical_record_graph(self)

    async def process(
        self,
        record: dict[str, Any],
        *,
        use_rag: bool = False,
        stage_names: list[str] | tuple[str, ...] | None = None,
        skip_record_filter: bool = False,
    ) -> dict[str, Any]:
        result = await self.workflow.ainvoke(
            {
                "record": record,
                "stage_names": stage_names,
                "use_rag": use_rag,
                "skip_record_filter": skip_record_filter,
            }
        )
        return result["record"]


def get_graph(config: RunnableConfig) -> Any:
    """Create a JSONL-file graph for LangGraph Server using environment model settings.

    The optional configurable values ``input_dir``, ``output_dir``,
    ``stage_timeout``, and ``enable_review`` can be supplied per assistant.
    Invocation input must include ``input_file`` with a path to a JSONL file.
    Model names, API keys, and base URLs use the same environment-backed defaults as
    the existing command-line runner.
    """
    configurable = config.get("configurable", {})
    args = record_agents.parse_args(argv=[])

    text_model = record_agents.build_model(
        args.model,
        args,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    diagnosis_model = record_agents.build_model(
        args.diagnosis_model,
        args,
        base_url=args.diagnosis_base_url,
        api_key=args.diagnosis_api_key,
        max_tokens=1000,
    )
    vlm_model = record_agents.build_model(
        args.vlm_model or args.model,
        args,
        base_url=args.vlm_base_url or args.base_url,
        api_key=args.vlm_api_key or args.api_key,
    )
    image_classification_model = record_agents.build_model(
        args.image_classification_model,
        args,
        base_url=args.image_classification_base_url or args.base_url,
        api_key=args.image_classification_api_key or args.api_key,
        max_tokens=6000,
    )

    enable_review = bool(configurable.get("enable_review", args.enable_review))
    reviewer_model = None
    if enable_review:
        reviewer_model = record_agents.build_model(
            args.reviewer_model,
            args,
            base_url=args.reviewer_base_url,
            api_key=args.reviewer_api_key,
        )

    default_input_dir = args.input if args.input.is_dir() else args.input.parent
    input_dir = Path(configurable.get("input_dir", default_input_dir))
    output_dir = Path(configurable.get("output_dir", args.output_dir))
    agents = LangGraphMedicalRecordAgents(
        text_model,
        vlm_model,
        input_dir,
        diagnosis_model=diagnosis_model,
        image_classification_model=image_classification_model,
        reviewer_model=reviewer_model,
        enable_review=enable_review,
        filter_path=output_dir / record_agents.FILTER_FILE_NAME,
        stage_timeout=float(configurable.get("stage_timeout", args.timeout)),
    )
    return build_medical_record_jsonl_graph(agents)


def main() -> None:
    """Run the existing file/MySQL CLI with the LangGraph record processor."""

    record_agents.main(
        agent_factory=LangGraphMedicalRecordAgents,
        description="基于 LangGraph 的多医生病历 AI 清洗与结构化脚本",
    )


if __name__ == "__main__":
    main()
