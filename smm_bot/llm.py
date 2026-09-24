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
import re
import time
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
    """POST /chat/completions — формат, который поддерживают почти все.

    endpoint можно задать целиком: некоторые сервисы (Pollinations) отдают
    chat completions не по /chat/completions, а по своему пути.
    Ключ может быть пустым — тогда заголовок авторизации не отправляем.
    """

    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 60,
                 endpoint: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint or f"{self.base_url}/chat/completions"
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMReply:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        # без лимита бесплатные модели иногда думают по минуте — обрезаем
        payload["max_tokens"] = 1200
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # бесплатный сервис регулярно отдаёт 5xx и обрывы — без повторов
        # пользователь видит «работаю без нейросети» на ровном месте
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.session.post(self.endpoint, headers=headers, json=payload,
                                         timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = LLMError(f"LLM недоступен: {exc}")
                time.sleep(0.8 * (attempt + 1))
                continue

            if resp.status_code >= 500:
                last_exc = LLMError(f"LLM вернул HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(0.8 * (attempt + 1))
                continue
            if resp.status_code == 429:
                # бесплатный сервис пускает по одному запросу на IP и отвечает
                # «Queue full». Секунда-две ожидания обычно решают вопрос —
                # без этого повтора владелец сразу видел «нейросеть не ответила».
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else 2.0 * (attempt + 1)
                except ValueError:
                    wait = 2.0 * (attempt + 1)
                last_exc = LLMError(f"LLM вернул HTTP 429 (очередь занята): {resp.text[:200]}")
                time.sleep(min(wait, 8.0))
                continue
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

        raise last_exc or LLMError("LLM недоступен")


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


POLLINATIONS_ENDPOINT = "https://text.pollinations.ai/openai"
POLLINATIONS_REFERRER = "smm-lead-bot"
# Порядок важен: пробуем по очереди, пока какая-нибудь модель не ответит.
# Без referrer у бесплатного режима сразу кончается квота анонимного ключа.
# "mistral" больше не существует в API (404), а "openai" лишь алиас openai-fast.
POLLINATIONS_MODELS = ["openai-fast", "openai"]
POLLINATIONS_MODEL = POLLINATIONS_MODELS[0]  # совместимость с прежним именем

# Бесплатный сервис дописывает в ответ рекламный блок. Владельцу он не нужен,
# а выглядит как часть ответа ассистента. Отрезаем всё от разделителя.
_AD_TAIL = re.compile(r"\n+\s*-{3,}\s*\n.*$", re.S)
_AD_LINE = re.compile(r"^\s*(?:\W*\s*)?(?:Support Pollinations|Sponsored|Ad\b|"
                      r"Powered by Pollinations)", re.I)


def strip_ads(text: str) -> str:
    """Убрать рекламный хвост бесплатной модели. Текст без рекламы не трогаем."""
    if not text:
        return text
    cleaned = _AD_TAIL.sub("", text)
    kept: list[str] = []
    for line in cleaned.splitlines():
        if _AD_LINE.match(line):
            break  # с этой строки начинается блок рекламы — дальше уже не ответ
        kept.append(line)
    return "\n".join(kept).strip()


class PollinationsLLM(BaseLLM):
    """Бесплатный режим без ключа.

    Два неочевидных момента, оба проверены на живом сервисе:
      * запрос без referrer отбивается сообщением "reached its budget",
        причём с HTTP 200 — поэтому отказ распознаём по тексту, а не по статусу;
      * не все модели умеют вызывать инструменты, поэтому перебираем список.
    """

    def __init__(self, models: list[str], timeout: int = 35) -> None:
        self.models = models
        self.timeout = timeout
        self.session = requests.Session()
        self.endpoint = f"{POLLINATIONS_ENDPOINT}?referrer={POLLINATIONS_REFERRER}"

    @staticmethod
    def _is_quota_refusal(reply: LLMReply) -> bool:
        low = reply.text.lower()
        return not reply.tool_calls and ("budget" in low or "rate limit" in low)

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             attempts: int = 2) -> LLMReply:
        last: LLMReply | None = None
        errors: list[str] = []
        for model in self.models:
            client = OpenAICompatibleLLM("", POLLINATIONS_ENDPOINT, model,
                                         timeout=self.timeout, endpoint=self.endpoint)
            for _ in range(attempts):
                try:
                    reply = client.chat(messages, tools)
                except LLMError as exc:
                    # сервис отдаёт 400/500 и на валидном запросе, поэтому пробуем ещё
                    errors.append(f"{model}: {exc}")
                    continue
                if reply.tool_calls or not self._is_quota_refusal(reply):
                    reply.text = strip_ads(reply.text)
                    return reply
                last = reply
                errors.append(f"{model}: отказ по квоте")
                break  # квота — не наш случай, сразу берём следующую модель
        if last is not None:
            last.text = strip_ads(last.text)
            return last
        raise LLMError("Бесплатная модель недоступна: " + "; ".join(errors))


def build_llm() -> BaseLLM:
    provider = (settings.llm_provider or "none").lower()

    # бесплатный режим: ключ не нужен вообще
    if provider == "pollinations":
        if settings.llm_model:
            return PollinationsLLM([settings.llm_model] + POLLINATIONS_MODELS)
        return PollinationsLLM(POLLINATIONS_MODELS)

    if provider == "none" or not settings.llm_api_key:
        return NoLLM()
    if provider == "anthropic":
        return AnthropicLLM(settings.llm_api_key, settings.llm_model or "claude-3-5-sonnet-latest")
    return OpenAICompatibleLLM(
        settings.llm_api_key,
        settings.llm_base_url or "https://api.openai.com/v1",
        settings.llm_model or "gpt-4o-mini",
    )
