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

import asyncio
from types import SimpleNamespace

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from medical.data_utils import structured_agent


class ExampleResult(BaseModel):
    value: str


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.bind_kwargs = None
        self.structured_kwargs = None

    def bind(self, **kwargs):
        self.bind_kwargs = kwargs
        return self

    def with_structured_output(self, schema, **kwargs):
        self.structured_kwargs = (schema, kwargs)
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self.responses.pop(0)


def test_tool_agent_builds_langchain_agent_and_validates_output(monkeypatch) -> None:
    calls = []

    class FakeGraph:
        async def ainvoke(self, payload):
            calls.append(payload)
            return {
                "structured_response": {"value": "tool"},
                "messages": [SimpleNamespace(content="tool response", tool_calls=[])],
            }

    created = {}

    def fake_create_agent(**kwargs):
        created.update(kwargs)
        return FakeGraph()

    monkeypatch.setattr(structured_agent, "create_agent", fake_create_agent)
    agent = structured_agent.StructuredAgent(
        object(),
        ExampleResult,
        system_prompt="提取字段。",
        name="example_agent",
    )

    result = asyncio.run(agent.ainvoke(HumanMessage(content="输入内容")))

    assert result.output == ExampleResult(value="tool")
    assert "tool response" in result.trace
    assert created["name"] == "example_agent"
    assert created["system_prompt"] == "提取字段。"
    assert calls[0]["messages"][0].content == "输入内容"


def test_model_config_forwards_generation_parameters(monkeypatch) -> None:
    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(structured_agent, "ChatOpenAI", fake_chat_openai)
    model = structured_agent.AgentModelConfig(
        model="qwen-plus",
        api_key="test-key",
        base_url="https://example.test/v1",
        temperature=0.6,
        thinking=True,
        max_tokens=2048,
        timeout=30,
        max_retries=4,
        client_kwargs={"top_p": 0.8},
        extra_body={"custom": "value"},
    ).build()

    assert model is not None
    assert captured == {
        "model": "qwen-plus",
        "api_key": "test-key",
        "base_url": "https://example.test/v1",
        "temperature": 0.6,
        "top_p": 0.8,
        "timeout": 30,
        "max_tokens": 2048,
        "max_retries": 4,
        "extra_body": {
            "custom": "value",
            "enable_thinking": True,
        },
    }


def test_json_object_agent_injects_schema_and_retries_validation_error() -> None:
    model = FakeModel(
        [
            SimpleNamespace(content="{}"),
            SimpleNamespace(content='{"value": "valid"}'),
        ]
    )
    agent = structured_agent.StructuredAgent(
        model,
        ExampleResult,
        system_prompt="提取字段。",
        output_mode=structured_agent.StructuredOutputMode.JSON_OBJECT,
    )

    result = asyncio.run(agent.ainvoke(HumanMessage(content="输入内容")))

    assert result.output == ExampleResult(value="valid")
    assert result.trace == '{"value": "valid"}'
    assert model.bind_kwargs == {"response_format": {"type": "json_object"}}
    assert len(model.calls) == 2
    assert "JSON Schema" in model.calls[0][0].content
    assert "无法通过结构化校验" in model.calls[1][-1].content


def test_json_schema_agent_returns_parsed_output_and_raw_trace() -> None:
    raw = SimpleNamespace(content='{"value":"native"}', tool_calls=[])
    model = FakeModel(
        [
            {
                "raw": raw,
                "parsed": ExampleResult(value="native"),
                "parsing_error": None,
            }
        ]
    )
    agent = structured_agent.StructuredAgent(
        model,
        ExampleResult,
        system_prompt="提取字段。",
        output_mode=structured_agent.StructuredOutputMode.JSON_SCHEMA,
    )

    result = asyncio.run(agent.ainvoke([HumanMessage(content="输入内容")]))

    assert result.output.value == "native"
    assert "native" in result.trace
    assert model.structured_kwargs == (
        ExampleResult,
        {"method": "json_schema", "include_raw": True},
    )
