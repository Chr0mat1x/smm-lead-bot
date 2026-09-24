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
        self.headers: dict = {}

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


def test_pollinations_uses_referrer() -> None:
    """Без referrer бесплатный режим отбивается сообщением про budget."""
    from smm_bot.llm import POLLINATIONS_REFERRER, PollinationsLLM

    llm = PollinationsLLM(["openai"])
    assert "referrer=" in llm.endpoint
    assert POLLINATIONS_REFERRER in llm.endpoint


def test_pollinations_retries_other_model_on_quota_refusal(monkeypatch) -> None:
    """Отказ по квоте приходит с HTTP 200 — распознаём по тексту и берём
    следующую модель, а не отдаём владельцу сообщение про бюджет."""
    from smm_bot.llm import LLMReply, PollinationsLLM

    calls: list[str] = []

    def fake_chat(self, messages, tools=None):
        calls.append(self.model)
        if self.model == "openai-fast":
            return LLMReply(text="The API key used for this request has reached its budget.")
        return LLMReply(text="работаю")

    monkeypatch.setattr("smm_bot.llm.OpenAICompatibleLLM.chat", fake_chat)
    reply = PollinationsLLM(["openai-fast", "openai"]).chat([{"role": "user", "content": "x"}])
    assert reply.text == "работаю"
    assert calls == ["openai-fast", "openai"]


def test_pollinations_retries_same_model_on_transient_error(monkeypatch) -> None:
    """Сервис отдаёт 400 и на корректном запросе — одну ошибку не считаем приговором."""
    from smm_bot.llm import LLMError, LLMReply, PollinationsLLM

    calls: list[str] = []

    def fake_chat(self, messages, tools=None):
        calls.append(self.model)
        if len(calls) == 1:
            raise LLMError("LLM вернул HTTP 400: Bad Request")
        return LLMReply(text="со второй попытки получилось")

    monkeypatch.setattr("smm_bot.llm.OpenAICompatibleLLM.chat", fake_chat)
    reply = PollinationsLLM(["openai-fast"]).chat([{"role": "user", "content": "x"}])
    assert reply.text == "с второй попытки получилось" or "второй" in reply.text
    assert len(calls) == 2


def test_pollinations_keeps_tool_call_even_with_odd_text(monkeypatch) -> None:
    """Вызов инструмента — успех, что бы ни было в тексте рядом с ним."""
    from smm_bot.llm import LLMReply, PollinationsLLM, ToolCall

    def fake_chat(self, messages, tools=None):
        return LLMReply(text="budget", tool_calls=[ToolCall(id="1", name="stats", arguments={})])

    monkeypatch.setattr("smm_bot.llm.OpenAICompatibleLLM.chat", fake_chat)
    reply = PollinationsLLM(["openai"]).chat([{"role": "user", "content": "x"}])
    assert reply.wants_tools
