"""LLM-клиент для диалога и генерации сообщений.

Поддерживает три режима, чтобы вы могли выбрать по вкусу и бюджету:

  * openai    — любой OpenAI-совместимый endpoint: OpenAI, OpenRouter, Groq,
                Together, vLLM, локальный Ollama (/v1). Задаётся LLM_BASE_URL.
  * anthropic — Claude через официальный API.
  * none      — ключа нет. Диалог работает в упрощённом режиме: бот отвечает
                по шаблонам и всё равно умеет править лиды. Нужен, чтобы бот
                не падал, пока вы не вписали ключ.

Наружу отдаём единый интерфейс chat(): список сообщений + описание инструментов
-> текст ответа или запрос на вызов инструмента.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import requests

from .config import settings

log = logging.getLogger("smm_bot.llm")


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class BaseLLM:
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    """POST /chat/completions — формат, который поддерживают почти все."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 60) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            resp = self.session.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(f"LLM недоступен: {exc}") from exc

        if resp.status_code != 200:
            raise LLMError(f"LLM вернул HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Неожиданный ответ LLM: {str(data)[:300]}") from exc

        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=raw.get("id", ""), name=fn.get("name", ""), arguments=args))
        return LLMReply(text=message.get("content") or "", tool_calls=calls)


class AnthropicLLM(BaseLLM):
    """Claude через /v1/messages. Формат инструментов у него свой."""

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.anthropic.com",
                 timeout: int = 60) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _split_system(self, messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        return "\n\n".join(system_parts), rest

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        system, rest = self._split_system(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1500,
            "messages": rest,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {"name": t["function"]["name"],
                 "description": t["function"].get("description", ""),
                 "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}})}
                for t in tools
            ]

        try:
            resp = self.session.post(
                f"{self.base_url}/v1/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(f"LLM недоступен: {exc}") from exc

        if resp.status_code != 200:
            raise LLMError(f"LLM вернул HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(id=block.get("id", ""), name=block.get("name", ""),
                                      arguments=block.get("input") or {}))
        return LLMReply(text="\n".join(text_parts).strip(), tool_calls=calls)


class NoLLM(BaseLLM):
    """Режим без ключа: бот работает, но без "разговора"."""

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        raise LLMError(
            "Ключ LLM не задан. Добавьте в .env: LLM_PROVIDER, LLM_API_KEY, LLM_MODEL."
        )


def build_llm() -> BaseLLM:
    provider = (settings.llm_provider or "none").lower()
    if provider == "none" or not settings.llm_api_key:
        return NoLLM()
    if provider == "anthropic":
        return AnthropicLLM(settings.llm_api_key, settings.llm_model or "claude-3-5-sonnet-latest")
    return OpenAICompatibleLLM(
        settings.llm_api_key,
        settings.llm_base_url or "https://api.openai.com/v1",
        settings.llm_model or "gpt-4o-mini",
    )
