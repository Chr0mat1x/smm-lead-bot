"""Разбор Telegram-контактов и проверка, что это канал, а не личный аккаунт.

В OSM ссылку на Telegram пишут как попало: «@name», «t.me/name»,
«https://t.me/+AbCdEf» (приглашение в закрытую группу), а иногда кладут её
в поле website, хотя сайта у бизнеса нет. Пока контакт не приведён к одному
виду, по нему нельзя ни дать ссылку, ни проверить, канал это или человек.

Проверка делается через Bot API getChat: бот может спросить у Telegram тип
чата по @username и не тратит на это права администратора. Личные аккаунты
и мёртвые юзернеймы отсеиваются — по ним нельзя написать от лица канала.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import requests

log = logging.getLogger("smm_bot.telegram")

TG_HOSTS = ("t.me", "telegram.me", "telegram.dog")
URL_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:t\.me|telegram\.me|telegram\.dog)/([^\s/?#]+)", re.I)
AT_RE = re.compile(r"(?<![A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_]{4,})")
BARE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{4,}$")
# t.me/+AbCdEf и t.me/joinchat/AbCdEf — приглашения, юзернейма нет
INVITE_RE = re.compile(r"^(?:\+|joinchat/)", re.I)
# тип чата в Bot API -> наш короткий код
KIND_BY_TYPE = {"channel": "channel", "supergroup": "group", "group": "group",
                "private": "private"}


@dataclass
class TelegramRef:
    """Приведённый Telegram-контакт.

    kind: channel | group | private | invite | unknown
      * channel — публичный канал, можно дать ссылку и написать в комментарии;
      * group   — супергруппа/чат, писать можно только если разрешено;
      * private — личный аккаунт, писать только вручную и осторожно;
      * invite  — закрытая ссылка-приглашение, юзернейма нет;
      * unknown — ещё не проверяли.
    """

    handle: str = ""       # без @, либо invite-хвост для ссылок-приглашений
    kind: str = "unknown"
    title: str = ""

    @property
    def is_invite(self) -> bool:
        return self.kind == "invite" or bool(INVITE_RE.match(self.handle))

    @property
    def is_public(self) -> bool:
        return self.kind in ("channel", "group") and not self.is_invite

    @property
    def is_channel(self) -> bool:
        return self.kind == "channel"

    @property
    def url(self) -> str:
        """Рабочая ссылка. Для приглашения — как есть, иначе t.me/@handle."""
        if not self.handle:
            return ""
        if self.is_invite:
            tail = self.handle.lstrip("+")
            return f"https://t.me/+{tail}"
        return f"https://t.me/{self.handle}"

    @property
    def at(self) -> str:
        return f"@{self.handle}" if self.handle and not self.is_invite else ""


def extract_handle(raw: str) -> str:
    """Достаём из любой записи юзернейм (без @) или хвост приглашения.

    Порядок важен: сначала ссылки (в них бывает «?» и слэши), потом @имя,
    и лишь в конце — «голый» юзернейм. Пустая строка означает «не разобрали».
    """
    text = (raw or "").strip()
    if not text:
        return ""

    match = URL_RE.search(text)
    if match:
        return match.group(1).strip()

    match = AT_RE.search(text)
    if match:
        return match.group(1).strip()

    # бывает «name1;name2» — берём первый осмысленный
    for chunk in re.split(r"[;,\s]+", text):
        chunk = chunk.strip().lstrip("@")
        if BARE_RE.match(chunk):
            return chunk
    return ""


def normalize(raw: str, kind: str = "unknown", title: str = "") -> TelegramRef:
    """Строка из OSM -> TelegramRef с уже известным типом (если проверяли)."""
    handle = extract_handle(raw)
    if not handle:
        return TelegramRef()
    if INVITE_RE.match(handle):
        return TelegramRef(handle=handle, kind="invite", title=title)
    if kind not in ("channel", "group", "private"):
        kind = "unknown"
    return TelegramRef(handle=handle, kind=kind, title=title)


def looks_like_telegram(value: str) -> bool:
    """Похоже ли значение на Telegram-ссылку.

    Нужно, чтобы отличить business с настоящим сайтом от бизнеса без сайта,
    который в поле website положил ссылку на свой канал.
    """
    low = (value or "").strip().lower()
    if not low:
        return False
    if any(host in low for host in TG_HOSTS):
        return True
    return bool(AT_RE.fullmatch(low.strip()))


def lookup(handle: str, token: str, timeout: int = 15) -> TelegramRef:
    """Спросить у Telegram тип чата по юзернейму.

    Бот видит публичные каналы и группы, даже не состоя в них. «chat not found»
    означает, что юзернейм занят личным аккаунтом или не существует — для нашей
    задачи это одно и то же: публичного канала нет.

    Ответ «Too Many Requests» (429) и ошибки сети — это НЕ «канала нет»:
    при частых проверках Telegram временно блокирует запросы, и такие контакты
    остаются непроверенными (unknown), чтобы проверить их позже. Спутать 429
    с «не найден» — значит навсегда потерять настоящий канал.
    """
    if not handle or not token:
        return TelegramRef()
    if INVITE_RE.match(handle):
        return TelegramRef(handle=handle, kind="invite")
    try:
        resp = requests.get(f"https://api.telegram.org/bot{token}/getChat",
                            params={"chat_id": f"@{handle}"}, timeout=timeout)
    except requests.RequestException as exc:
        log.debug("getChat @%s не ответил: %s", handle, exc)
        return TelegramRef(handle=handle, kind="unknown")

    if resp.status_code == 429:
        log.warning("getChat упёрся в лимит Telegram на @%s — проверим позже", handle)
        return TelegramRef(handle=handle, kind="unknown")
    if resp.status_code >= 500:
        log.debug("getChat @%s вернул HTTP %s", handle, resp.status_code)
        return TelegramRef(handle=handle, kind="unknown")

    data = resp.json() if resp.content else {}
    if not data.get("ok"):
        description = (data.get("description") or "").lower()
        # 429 и «retry» приходят и как HTTP 200; отличаем их от реального отсутствия
        if "too many requests" in description or "retry" in description:
            log.warning("getChat @%s: лимит Telegram (%s) — проверим позже", handle, description)
            return TelegramRef(handle=handle, kind="unknown")
        return TelegramRef(handle=handle, kind="not_found")
    result = data["result"]
    return TelegramRef(handle=handle,
                       kind=KIND_BY_TYPE.get(result.get("type", ""), "unknown"),
                       title=result.get("title") or "")
