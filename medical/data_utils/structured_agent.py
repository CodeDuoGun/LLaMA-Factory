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

"""Reusable LangChain model configuration and structured Agent invocation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeVar

from langchain.agents import create_agent
from langchain.agents.structured_output import StructuredOutputError, ToolStrategy
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class StructuredOutputMode(StrEnum):
    """How a provider is asked to generate structured output."""

    TOOL = "tool"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


@dataclass(slots=True)
class AgentModelConfig:
    """Provider-independent model parameters for one or more Agents."""

    model: str
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    thinking: bool | None = False
    max_tokens: int | None = None
    timeout: float | None = 120.0
    max_retries: int = 2
    client_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    def build(self) -> ChatOpenAI:
        """Build a ChatOpenAI client while preserving provider-specific thinking syntax."""
        kwargs: dict[str, Any] = dict(self.client_kwargs)
        kwargs.update(
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_retries": self.max_retries,
            }
        )
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        extra_body = dict(self.extra_body)
        if self.thinking is not None and not _has_thinking_setting(extra_body):
            provider_hint = f"{self.model} {self.base_url}".lower()
            if "deepseek" in provider_hint:
                extra_body["thinking"] = {"type": "enabled" if self.thinking else "disabled"}
            else:
                extra_body["enable_thinking"] = self.thinking
        if extra_body:
            kwargs["extra_body"] = extra_body
        return ChatOpenAI(**kwargs)


@dataclass(slots=True)
class StructuredAgentResult(Generic[SchemaT]):
    """Validated output plus an observable response trace for auditing."""

    output: SchemaT
    trace: str = ""


class StructuredAgent(Generic[SchemaT]):
    """Invoke one Agent and normalize provider output into a validated Pydantic model."""

    def __init__(
        self,
        model: Any,
        schema: type[SchemaT],
        *,
        system_prompt: str = "",
        name: str = "",
        output_mode: StructuredOutputMode | str = StructuredOutputMode.TOOL,
        validation_retries: int = 1,
    ) -> None:
        self.model = model
        self.schema = schema
        self.system_prompt = system_prompt
        self.name = name or schema.__name__
        self.output_mode = StructuredOutputMode(output_mode)
        self.validation_retries = max(0, validation_retries)
        self.runnable = self._build_runnable()

    @classmethod
    def from_config(
        cls,
        config: AgentModelConfig,
        schema: type[SchemaT],
        **kwargs: Any,
    ) -> StructuredAgent[SchemaT]:
        """Build a model and structured Agent from a single reusable configuration."""
        return cls(config.build(), schema, **kwargs)

    def _build_runnable(self) -> Any:
        if self.output_mode == StructuredOutputMode.TOOL:
            return create_agent(
                model=self.model,
                tools=[],
                system_prompt=self.system_prompt,
                response_format=ToolStrategy(schema=self.schema, handle_errors=True),
                name=self.name,
            )
        if self.output_mode == StructuredOutputMode.JSON_OBJECT:
            return self.model.bind(response_format={"type": "json_object"})
        return self.model.with_structured_output(
            self.schema,
            method="json_schema",
            include_raw=True,
        )

    async def ainvoke(
        self,
        messages: BaseMessage | Sequence[BaseMessage],
    ) -> StructuredAgentResult[SchemaT]:
        """Invoke asynchronously, retry formatting failures, and return validated output."""
        source_messages = [messages] if isinstance(messages, BaseMessage) else list(messages)
        last_error: Exception | None = None
        for attempt in range(self.validation_retries + 1):
            attempt_messages = list(source_messages)
            if attempt:
                attempt_messages.append(self._retry_message())
            try:
                response = await self._invoke_once(attempt_messages)
                return self._format_response(response)
            except (json.JSONDecodeError, StructuredOutputError, ValidationError, ValueError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"{self.name} 未返回 {self.schema.__name__} 结构化结果")

    async def _invoke_once(self, messages: list[BaseMessage]) -> Any:
        if self.output_mode == StructuredOutputMode.TOOL:
            return await self.runnable.ainvoke({"messages": messages})
        prompt_messages: list[BaseMessage] = []
        if self.system_prompt:
            prompt_messages.append(SystemMessage(content=self._json_system_prompt()))
        prompt_messages.extend(messages)
        return await self.runnable.ainvoke(prompt_messages)

    def _format_response(self, response: Any) -> StructuredAgentResult[SchemaT]:
        if self.output_mode == StructuredOutputMode.TOOL:
            if not isinstance(response, Mapping):
                raise ValueError(f"{self.name} 未返回 Agent 状态")
            candidate = response.get("structured_response")
            trace = response_trace(response)
        elif self.output_mode == StructuredOutputMode.JSON_SCHEMA:
            if isinstance(response, Mapping) and "parsed" in response:
                parsing_error = response.get("parsing_error")
                if parsing_error is not None:
                    raise ValueError(str(parsing_error))
                candidate = response.get("parsed")
                trace = message_trace(response.get("raw"))
            else:
                candidate = response
                trace = message_trace(response)
        else:
            raw_text = response_text(response)
            candidate = json.loads(_strip_json_fence(raw_text))
            trace = raw_text

        if isinstance(candidate, self.schema):
            output = candidate
        else:
            output = self.schema.model_validate(candidate)
        return StructuredAgentResult(output=output, trace=trace)

    def _json_system_prompt(self) -> str:
        if self.output_mode != StructuredOutputMode.JSON_OBJECT:
            return self.system_prompt
        schema_json = json.dumps(model_json_schema(self.schema), ensure_ascii=False)
        return (
            f"{self.system_prompt}\n"
            "必须仅返回一个完整、合法的 JSON 对象，不要输出 Markdown、注释或额外说明。"
            f"\nJSON Schema：{schema_json}"
        ).strip()

    def _retry_message(self) -> HumanMessage:
        if self.output_mode == StructuredOutputMode.TOOL:
            requirement = "只调用一次结构化输出工具，不得重复调用"
        else:
            requirement = "只返回完整 JSON 对象，不得输出 Markdown 或额外说明"
        return HumanMessage(
            content=(
                f"上一次 {self.schema.__name__} 输出无法通过结构化校验。请重新生成一次，"
                f"严格遵守字段名称和字段类型；{requirement}。"
            )
        )


def model_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Return a Pydantic v1/v2 compatible JSON Schema."""
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    return schema.schema()


def response_text(response: Any) -> str:
    """Normalize OpenAI-compatible text and content-block responses."""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                texts.append(str(block.get("text") or "").strip())
        return "\n".join(text for text in texts if text).strip()
    return str(content or "").strip()


def message_trace(message: Any) -> str:
    """Serialize one observable model message without hidden reasoning."""
    if message is None:
        return ""
    entry = {
        "message_type": type(message).__name__,
        "content": _trace_value(getattr(message, "content", "")),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = _trace_value(tool_calls)
    invalid_tool_calls = getattr(message, "invalid_tool_calls", None)
    if invalid_tool_calls:
        entry["invalid_tool_calls"] = _trace_value(invalid_tool_calls)
    return json.dumps([entry], ensure_ascii=False, default=str)


def response_trace(response: Any) -> str:
    """Serialize observable AI messages from a LangGraph Agent response."""
    if not isinstance(response, Mapping):
        return response_text(response)
    trace = []
    for message in response.get("messages") or []:
        if type(message).__name__ in {"HumanMessage", "SystemMessage"}:
            continue
        entry = {
            "message_type": type(message).__name__,
            "content": _trace_value(getattr(message, "content", "")),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = _trace_value(tool_calls)
        invalid_tool_calls = getattr(message, "invalid_tool_calls", None)
        if invalid_tool_calls:
            entry["invalid_tool_calls"] = _trace_value(invalid_tool_calls)
        trace.append(entry)
    return json.dumps(trace, ensure_ascii=False, default=str) if trace else ""


def _trace_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return value.dict()
    if isinstance(value, Mapping):
        return {str(key): _trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trace_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _strip_json_fence(text: str) -> str:
    if text.startswith("```") and text.endswith("```"):
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    return text


def _has_thinking_setting(extra_body: Mapping[str, Any]) -> bool:
    return "enable_thinking" in extra_body or "thinking" in extra_body or "chat_template_kwargs" in extra_body
