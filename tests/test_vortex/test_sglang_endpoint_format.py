import os

import pytest
import requests

from tau2.config import get_nl_assertions_llm_config

SGLANG_BASE_URL_ENV_VAR = "SGLANG_OPENAI_BASE_URL"
SGLANG_MODEL_ENV_VAR = "SGLANG_OPENAI_MODEL"
SGLANG_API_KEY_ENV_VAR = "SGLANG_OPENAI_API_KEY"


def _sglang_base_url() -> str:
    return os.getenv(SGLANG_BASE_URL_ENV_VAR, "http://127.0.0.1:30000/v1")


def _sglang_api_key() -> str:
    return os.getenv(SGLANG_API_KEY_ENV_VAR, "sk-local")


def _sglang_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_sglang_api_key()}",
    }


def _ensure_live_sglang() -> str:
    base_url = _sglang_base_url()
    try:
        response = requests.get(
            f"{base_url}/models",
            headers=_sglang_headers(),
            timeout=5,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"Local SGLang endpoint not reachable at {base_url}: {exc}")

    payload = response.json()
    models = payload.get("data") or []
    if not models:
        pytest.skip(f"No models returned by local SGLang endpoint at {base_url}")

    return os.getenv(SGLANG_MODEL_ENV_VAR, models[0]["id"])


def _post_chat_completion(payload: dict) -> dict:
    response = requests.post(
        f"{_sglang_base_url()}/chat/completions",
        headers=_sglang_headers(),
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def test_nl_assertions_judge_can_use_local_env_override(monkeypatch):
    monkeypatch.setenv("TAU2_LLM_NL_ASSERTIONS_MODEL", "openai/Qwen/Qwen3-32B")
    monkeypatch.setenv(
        "TAU2_LLM_NL_ASSERTIONS_API_BASE", "http://127.0.0.1:30000/v1"
    )
    monkeypatch.setenv("TAU2_LLM_NL_ASSERTIONS_API_KEY", "sk-local")
    monkeypatch.setenv("TAU2_LLM_NL_ASSERTIONS_TEMPERATURE", "0.0")
    monkeypatch.setenv(
        "TAU2_LLM_NL_ASSERTIONS_ARGS",
        '{"extra_body":{"chat_template_kwargs":{"enable_thinking":true},"separate_reasoning":true}}',
    )

    model, llm_args = get_nl_assertions_llm_config()

    assert model == "openai/Qwen/Qwen3-32B"
    assert llm_args["api_base"] == "http://127.0.0.1:30000/v1"
    assert llm_args["api_key"] == "sk-local"
    assert llm_args["temperature"] == 0.0
    assert llm_args["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert llm_args["extra_body"]["separate_reasoning"] is True


def test_live_sglang_returns_separate_reasoning_content():
    model = _ensure_live_sglang()
    payload = _post_chat_completion(
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "How many r's are in strawberry?"}
            ],
            "temperature": 0,
            "max_tokens": 256,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
                "separate_reasoning": True,
            },
        }
    )
    message = payload["choices"][0]["message"]

    assert "reasoning_content" in message, payload
    assert message["reasoning_content"], payload
    assert "content" in message, payload


def test_live_sglang_returns_parsed_tool_calls():
    model = _ensure_live_sglang()
    payload = _post_chat_completion(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "Call the tool before answering."},
                {
                    "role": "user",
                    "content": "Use the tool get_time now. Do not answer directly.",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Return the current time.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                }
            ],
            "tool_choice": "required",
            "temperature": 0,
            "max_tokens": 128,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
                "separate_reasoning": True,
            },
        }
    )
    choice = payload["choices"][0]
    message = choice["message"]

    assert choice["finish_reason"] == "tool_calls", payload
    assert "tool_calls" in message, payload
    assert len(message["tool_calls"]) == 1, payload
    assert message["tool_calls"][0]["function"]["name"] == "get_time", payload
    assert message["tool_calls"][0]["function"]["arguments"] == "{}", payload


def test_live_sglang_ignores_input_reasoning_content_on_normal_replay():
    model = _ensure_live_sglang()
    base_messages = [
        {"role": "system", "content": "Answer with one short sentence."},
        {"role": "user", "content": "Say the word alpha."},
        {"role": "assistant", "content": "alpha"},
        {"role": "user", "content": "Repeat the assistant's last answer exactly."},
    ]

    with_reasoning_messages = [
        {"role": "system", "content": "Answer with one short sentence."},
        {"role": "user", "content": "Say the word alpha."},
        {
            "role": "assistant",
            "content": "alpha",
            "reasoning_content": "hidden reasoning that should not affect replay",
        },
        {"role": "user", "content": "Repeat the assistant's last answer exactly."},
    ]

    common_payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 64,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
            "separate_reasoning": True,
        },
    }

    payload_without = _post_chat_completion(
        {**common_payload, "messages": base_messages}
    )
    payload_with = _post_chat_completion(
        {**common_payload, "messages": with_reasoning_messages}
    )

    assert payload_without["usage"]["prompt_tokens"] == payload_with["usage"][
        "prompt_tokens"
    ], (payload_without, payload_with)


def test_live_sglang_preserves_input_reasoning_content_on_generated_tool_loop_replay():
    model = _ensure_live_sglang()
    common_payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 512,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
            "separate_reasoning": True,
        },
    }
    first_turn_messages = [
        {
            "role": "system",
            "content": "Use the get_four tool first. After the tool result, answer with exactly the word four.",
        },
        {
            "role": "user",
            "content": "Use the tool get_four now. Do not answer directly.",
        },
    ]
    first_turn_payload = _post_chat_completion(
        {
            **common_payload,
            "messages": first_turn_messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_four",
                        "description": "Return the word four.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        }
    )
    first_turn_message = first_turn_payload["choices"][0]["message"]
    assert first_turn_payload["choices"][0]["finish_reason"] == "tool_calls", (
        first_turn_payload
    )
    assert first_turn_message.get("reasoning_content"), first_turn_payload
    assert first_turn_message.get("tool_calls"), first_turn_payload
    tool_call = first_turn_message["tool_calls"][0]
    assistant_tool_call_with_reasoning = {
        "role": "assistant",
        "content": first_turn_message.get("content"),
        "reasoning_content": first_turn_message.get("reasoning_content"),
        "tool_calls": first_turn_message["tool_calls"],
    }
    assistant_tool_call_without_reasoning = {
        "role": "assistant",
        "content": first_turn_message.get("content"),
        "tool_calls": first_turn_message["tool_calls"],
    }
    second_turn_base = [
        *first_turn_messages,
        assistant_tool_call_without_reasoning,
        {"role": "tool", "tool_call_id": tool_call["id"], "content": "four"},
    ]
    second_turn_with_reasoning = [
        *first_turn_messages,
        assistant_tool_call_with_reasoning,
        {"role": "tool", "tool_call_id": tool_call["id"], "content": "four"},
    ]

    payload_without = _post_chat_completion(
        {**common_payload, "max_tokens": 128, "messages": second_turn_base}
    )
    payload_with = _post_chat_completion(
        {**common_payload, "max_tokens": 128, "messages": second_turn_with_reasoning}
    )

    assert payload_with["usage"]["prompt_tokens"] > payload_without["usage"][
        "prompt_tokens"
    ], (payload_without, payload_with)
