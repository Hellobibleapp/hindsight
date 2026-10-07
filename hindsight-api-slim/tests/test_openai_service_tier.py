import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

TOOLS = [
    {
        "type": "function",
        "function": {"name": "recall", "parameters": {"type": "object", "properties": {}}},
    }
]


def _response(tool_call: bool = False) -> MagicMock:
    response = MagicMock()
    # A bare MagicMock auto-creates .error and model_dump(), which read as a provider error payload.
    response.error = None
    response.model_dump.return_value = {}
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 10
    response.usage.total_tokens = 110
    response.usage.completion_tokens_details = None
    response.usage.prompt_tokens_details = None
    message = response.choices[0].message
    message.refusal = None
    if tool_call:
        call = MagicMock()
        call.id = "call_1"
        call.function.name = "recall"
        call.function.arguments = json.dumps({})
        response.choices[0].finish_reason = "tool_calls"
        message.content = None
        message.tool_calls = [call]
    else:
        response.choices[0].finish_reason = "stop"
        message.content = "ok"
        message.tool_calls = None
    return response


@pytest.mark.asyncio
async def test_openai_service_tier_sent_on_both_paths():
    llm = OpenAICompatibleLLM(
        provider="openai", api_key="sk-test", base_url=None, model="gpt-5.6", openai_service_tier="flex"
    )
    messages = [{"role": "user", "content": "go"}]
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _response()
        await llm.call(messages=messages, max_retries=0)
        assert create.call_args.kwargs["service_tier"] == "flex"

        create.return_value = _response(tool_call=True)
        await llm.call_with_tools(messages=messages, tools=TOOLS, max_retries=0)
        assert create.call_args.kwargs["service_tier"] == "flex"


@pytest.mark.asyncio
async def test_openai_service_tier_not_sent_to_other_providers():
    llm = OpenAICompatibleLLM(
        provider="lmstudio",
        api_key="sk-test",
        base_url="http://localhost:1234/v1",
        model="openai/gpt-oss-20b",
        openai_service_tier="flex",
    )
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = _response()
        await llm.call(messages=[{"role": "user", "content": "go"}], max_retries=0)

    assert "service_tier" not in create.call_args.kwargs
    assert "service_tier" not in create.call_args.kwargs.get("extra_body", {})
