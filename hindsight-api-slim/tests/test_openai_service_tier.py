"""``HINDSIGHT_API_LLM_OPENAI_SERVICE_TIER`` must reach the wire on the chat-completions path.

The value was threaded from config into the provider (#2438) and stored on the instance, but
only ``groq_service_tier`` was ever written into a request — so on provider ``openai`` the knob
was a silent no-op and every call billed at standard rates instead of flex (half of standard on
input, cached input, cache write and output alike).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Recall semantic memories",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    }
]


def _make_llm(**kwargs) -> OpenAICompatibleLLM:
    params = {"provider": "openai", "api_key": "sk-test", "base_url": None, "model": "gpt-5.6-luna"}
    params.update(kwargs)
    return OpenAICompatibleLLM(**params)


def _text_response() -> MagicMock:
    response = MagicMock()
    # A bare MagicMock auto-creates .error and a MagicMock model_dump(), which the response
    # hardening reads as a provider error payload.
    response.error = None
    response.model_dump.return_value = {}
    response.usage.prompt_tokens = 3100
    response.usage.completion_tokens = 120
    response.usage.total_tokens = 3220
    response.usage.completion_tokens_details = None
    response.usage.prompt_tokens_details = None
    response.choices[0].finish_reason = "stop"
    response.choices[0].message.content = "ok"
    response.choices[0].message.tool_calls = None
    response.choices[0].message.refusal = None
    return response


def _tool_call_response() -> MagicMock:
    tool_call = MagicMock()
    tool_call.id = "call_abc123"
    tool_call.function.name = "recall"
    tool_call.function.arguments = json.dumps({"query": "who is the user"})

    response = _text_response()
    response.choices[0].finish_reason = "tool_calls"
    response.choices[0].message.content = None
    response.choices[0].message.tool_calls = [tool_call]
    return response


async def _capture_call(llm: OpenAICompatibleLLM) -> dict:
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _text_response()
        await llm.call(messages=[{"role": "user", "content": "go"}], max_retries=0)
        return mock_create.call_args.kwargs


async def _capture_tool_call(llm: OpenAICompatibleLLM) -> dict:
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _tool_call_response()
        await llm.call_with_tools(messages=[{"role": "user", "content": "go"}], tools=TOOLS, max_retries=0)
        return mock_create.call_args.kwargs


class TestOpenAIServiceTierReachesTheRequest:
    @pytest.mark.asyncio
    async def test_call_sends_the_tier(self):
        kwargs = await _capture_call(_make_llm(openai_service_tier="flex"))

        assert kwargs["service_tier"] == "flex"

    @pytest.mark.asyncio
    async def test_call_with_tools_sends_the_tier(self):
        kwargs = await _capture_tool_call(_make_llm(openai_service_tier="flex"))

        assert kwargs["service_tier"] == "flex"

    @pytest.mark.asyncio
    async def test_tier_is_top_level_not_extra_body(self):
        """Chat completions types ``service_tier`` natively; extra_body would be a second, ignored copy."""
        kwargs = await _capture_call(_make_llm(openai_service_tier="flex"))

        assert "service_tier" not in kwargs.get("extra_body", {})


class TestUnsetAndOtherProviders:
    @pytest.mark.asyncio
    async def test_unset_tier_omits_the_field(self):
        kwargs = await _capture_call(_make_llm())

        assert "service_tier" not in kwargs

    @pytest.mark.asyncio
    async def test_other_providers_do_not_receive_the_openai_tier(self):
        kwargs = await _capture_call(
            _make_llm(provider="lmstudio", base_url="http://localhost:1234/v1", openai_service_tier="flex")
        )

        assert "service_tier" not in kwargs
        assert "service_tier" not in kwargs.get("extra_body", {})

    @pytest.mark.asyncio
    async def test_groq_tier_still_rides_extra_body(self):
        llm = _make_llm(provider="groq", model="llama-3.3-70b", groq_service_tier="flex")
        kwargs = await _capture_call(llm)

        assert kwargs["extra_body"]["service_tier"] == "flex"
        assert "service_tier" not in kwargs
