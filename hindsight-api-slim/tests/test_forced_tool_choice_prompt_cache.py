"""Forced turns must not rewrite the tools array on backends that support named tool_choice.

The tools array is serialized ahead of the messages in a server-side prompt-cache prefix.
The reflect loop forces a different tool on each of its first iterations, so narrowing
``tools`` to the forced entry gives every iteration a different prefix and the shared system
prompt is never reused — measured as a 0% cached-token rate on the mental-model refresh lane
against native OpenAI, while retain and consolidation (which send no tools) cached normally.

Keeping the full tools array and forcing by name instead caches from the second call on.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.llm_interface import LLMToolChoice
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

REFLECT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    for name, desc in (
        ("search_mental_models", "Search consolidated mental models"),
        ("search_observations", "Search raw observations"),
        ("recall", "Recall semantic memories"),
        ("done", "Finish and return the answer"),
    )
]

# The forced sequence reflect/agent.py walks before it allows auto.
FORCED_SEQUENCE = ["search_mental_models", "search_observations", "recall"]


def _make_llm(**kwargs) -> OpenAICompatibleLLM:
    params = {"provider": "openai", "api_key": "sk-test", "base_url": None, "model": "gpt-5.6-luna"}
    params.update(kwargs)
    return OpenAICompatibleLLM(**params)


def _tool_call_response(tool_name: str) -> MagicMock:
    mock_tc = MagicMock()
    mock_tc.id = "call_abc123"
    mock_tc.function.name = tool_name
    mock_tc.function.arguments = json.dumps({"query": "who is the user"})

    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 2500
    mock_response.usage.completion_tokens = 40
    mock_response.usage.total_tokens = 2540
    mock_response.usage.completion_tokens_details = None
    mock_response.usage.prompt_tokens_details = None
    mock_response.choices[0].finish_reason = "tool_calls"
    mock_response.choices[0].message.content = None
    mock_response.choices[0].message.tool_calls = [mock_tc]
    return mock_response


async def _capture_call(llm: OpenAICompatibleLLM, forced_name: str) -> dict:
    """Run one forced call and return the kwargs that reached the OpenAI client."""
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _tool_call_response(forced_name)
        await llm.call_with_tools(
            messages=[{"role": "system", "content": "x" * 4000}, {"role": "user", "content": "go"}],
            tools=REFLECT_TOOLS,
            tool_choice=LLMToolChoice.named(forced_name),
            max_retries=0,
        )
        return mock_create.call_args.kwargs


class TestNamedToolChoiceKeepsToolsStable:
    @pytest.mark.asyncio
    async def test_native_openai_keeps_the_full_tools_array(self):
        kwargs = await _capture_call(_make_llm(), "search_mental_models")

        assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "search_mental_models"}}
        assert len(kwargs["tools"]) == len(REFLECT_TOOLS)
        assert [t["function"]["name"] for t in kwargs["tools"]] == [t["function"]["name"] for t in REFLECT_TOOLS]

    @pytest.mark.asyncio
    async def test_tools_payload_is_byte_identical_across_a_forced_sequence(self):
        """The cache-prefix property: only tool_choice may differ between forced turns."""
        llm = _make_llm()
        payloads = []
        for name in FORCED_SEQUENCE:
            kwargs = await _capture_call(llm, name)
            payloads.append(json.dumps(kwargs["tools"], sort_keys=True))

        assert len(set(payloads)) == 1, "tools array changed between forced turns; prompt cache would miss"

    @pytest.mark.asyncio
    async def test_azure_openai_host_also_keeps_the_full_array(self):
        llm = _make_llm(base_url="https://example.openai.azure.com/openai/v1")
        kwargs = await _capture_call(llm, "recall")

        assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "recall"}}
        assert len(kwargs["tools"]) == len(REFLECT_TOOLS)


class TestNarrowingPathPreserved:
    """Everything not on the allowlist keeps the portable narrow-to-one-tool behaviour."""

    @pytest.mark.asyncio
    async def test_unknown_openai_compatible_host_still_narrows(self):
        llm = _make_llm(base_url="https://api.x.ai/v1", model="grok-4.5")
        kwargs = await _capture_call(llm, "search_observations")

        assert kwargs["tool_choice"] == "required"
        assert [t["function"]["name"] for t in kwargs["tools"]] == ["search_observations"]

    @pytest.mark.asyncio
    async def test_deepseek_model_still_narrows(self):
        llm = _make_llm(model="deepseek-chat")
        kwargs = await _capture_call(llm, "recall")

        # DeepSeek rejects named/required tool_choice, so it is omitted entirely
        # and the narrowed tools list is what keeps the call forced.
        assert "tool_choice" not in kwargs
        assert [t["function"]["name"] for t in kwargs["tools"]] == ["recall"]

    @pytest.mark.asyncio
    async def test_lmstudio_still_narrows_and_downgrades(self):
        llm = _make_llm(provider="lmstudio", base_url="http://localhost:1234/v1", model="openai/gpt-oss-20b")
        kwargs = await _capture_call(llm, "search_mental_models")

        assert "tool_choice" not in kwargs
        assert [t["function"]["name"] for t in kwargs["tools"]] == ["search_mental_models"]


class TestCapabilityCheck:
    def test_allowlist(self):
        assert _make_llm()._supports_named_tool_choice() is True
        assert _make_llm(base_url="https://api.openai.com/v1")._supports_named_tool_choice() is True
        assert _make_llm(base_url="https://x.openai.azure.com")._supports_named_tool_choice() is True
        assert _make_llm(base_url="https://api.x.ai/v1")._supports_named_tool_choice() is False
        assert (
            _make_llm(provider="lmstudio", base_url="http://localhost:1234/v1")._supports_named_tool_choice() is False
        )
        assert _make_llm(model="deepseek-chat")._supports_named_tool_choice() is False

    def test_suffix_matching_is_not_substring_matching(self):
        assert _make_llm(base_url="https://openai.com.evil.example/v1")._supports_named_tool_choice() is False
