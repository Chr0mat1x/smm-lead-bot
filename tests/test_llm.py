"""Тесты LLM-клиентов: конструирование, заголовки, разбор ответов.

Сеть здесь не нужна: проверяем, что клиент правильно собирает запрос
(в частности, что при бесплатном режиме не шлёт пустой Authorization)
и корректно разбирает ответы. Живая проверка — в scripts/free_llm_demo.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smm_bot.llm import (OpenAICompatibleLLM, POLLINATIONS_ENDPOINT,  # noqa: E402
                         POLLINATIONS_MODEL, LLMError)


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self) -> dict:
        return self._payload


def capture(client: OpenAICompatibleLLM, payload: dict, status_code: int = 200) -> dict:
    """Подменяем сессию, чтобы поймать реальный сформированный запрос."""
    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        captured["url"] = url
        captured["headers"] = headers or {}
        captured["payload"] = json or {}
        return FakeResponse(payload, status_code)

    client.session.post = fake_post  # type: ignore[method-assign]
    return captured


def test_default_endpoint_is_appended() -> None:
    client = OpenAICompatibleLLM("key", "https://api.example.com/v1", "m")
    assert client.endpoint == "https://api.example.com/v1/chat/completions"


def test_custom_endpoint_is_used_as_is() -> None:
    """Pollinations отдаёт chat completions по своему пути, не /chat/completions."""
    client = OpenAICompatibleLLM("", POLLINATIONS_ENDPOINT, POLLINATIONS_MODEL,
                                 endpoint=POLLINATIONS_ENDPOINT)
    assert client.endpoint == "https://text.pollinations.ai/openai"


def test_free_mode_sends_no_auth_header() -> None:
    client = OpenAICompatibleLLM("", POLLINATIONS_ENDPOINT, POLLINATIONS_MODEL,
                                 endpoint=POLLINATIONS_ENDPOINT)
    captured = capture(client, {"choices": [{"message": {"content": "ок"}}]})
    client.chat([{"role": "user", "content": "привет"}])
    assert "Authorization" not in captured["headers"]


def test_paid_mode_sends_bearer() -> None:
    client = OpenAICompatibleLLM("secret", "https://api.example.com/v1", "m")
    captured = capture(client, {"choices": [{"message": {"content": "ок"}}]})
    client.chat([{"role": "user", "content": "привет"}])
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_tools_are_passed_and_parsed() -> None:
    client = OpenAICompatibleLLM("k", "https://api.example.com/v1", "m")
    payload = {"choices": [{"message": {
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "reject_lead",
                                     "arguments": '{"lead_key": "osm:node/1"}'}}],
    }}]}
    captured = capture(client, payload)
    tools = [{"type": "function", "function": {"name": "reject_lead", "parameters": {}}}]
    reply = client.chat([{"role": "user", "content": "x"}], tools=tools)
    assert captured["payload"]["tool_choice"] == "auto"
    assert reply.tool_calls[0].arguments["lead_key"] == "osm:node/1"


def test_malformed_tool_arguments_do_not_crash() -> None:
    """Модель может вернуть битый JSON в аргументах — не падаем."""
    client = OpenAICompatibleLLM("k", "https://api.example.com/v1", "m")
    payload = {"choices": [{"message": {
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "stats", "arguments": "{не json"}}],
    }}]}
    capture(client, payload)
    reply = client.chat([{"role": "user", "content": "x"}])
    assert reply.tool_calls[0].arguments == {}


def test_http_error_raises_llm_error() -> None:
    client = OpenAICompatibleLLM("k", "https://api.example.com/v1", "m")
    capture(client, {"error": "nope"}, status_code=429)
    with pytest.raises(LLMError, match="429"):
        client.chat([{"role": "user", "content": "x"}])
