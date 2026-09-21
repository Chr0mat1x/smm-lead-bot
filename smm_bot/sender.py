"""Полуручная отправка через Telethon (ваш личный Telegram-аккаунт).

ВАЖНО И ЧЕСТНО:
Telegram запрещает массовую рассылку незнакомым людям. Даже этот модуль,
который соблюдает паузы и дневной лимит, может привести к блокировке аккаунта,
если писать тем, кто вас не ждёт. Это не "обход защиты", а способ аккуратно
связаться с теми, с кем у вас уже есть контакт (общий чат, знакомый, заявка).

Правила, которые зашиты в код:
  * только лиды со статусом APPROVED, которых вы одобрили руками;
  * не больше DAILY_SEND_LIMIT сообщений в день (по умолчанию 15);
  * пауза не меньше MIN_SECONDS_BETWEEN_SENDS между сообщениями;
  * лиду, попросившему не писать, сообщение не уйдёт никогда;
  * каждый факт отправки пишется в журнал.

Запуск:  python -m smm_bot.sender --dry-run
         python -m smm_bot.sender
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone

from .config import settings
from .models import Channel, LeadStatus
from .storage import Storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("smm_bot.sender")


class LimitReached(RuntimeError):
    pass


def check_limits(storage: Storage) -> None:
    """Проверка дневного лимита и паузы между сообщениями."""
    sent = storage.sent_today()
    if sent >= settings.daily_send_limit:
        raise LimitReached(
            f"Дневной лимит исчерпан: {sent}/{settings.daily_send_limit}. "
            "Продолжите завтра — так аккаунт проживёт дольше."
        )
    last = storage.last_send_ts()
    if last:
        elapsed = (datetime.now(timezone.utc).replace(tzinfo=None) - last).total_seconds()
        if elapsed < settings.min_seconds_between_sends:
            wait = settings.min_seconds_between_sends - elapsed
            raise LimitReached(f"Слишком часто. Подождите ещё {int(wait)} сек.")


def _normalize_username(raw: str) -> str | None:
    value = (raw or "").strip()
    if not value:
        return None
    if value.startswith("https://t.me/"):
        value = value.removeprefix("https://t.me/")
    elif value.startswith("t.me/"):
        value = value.removeprefix("t.me/")
    value = value.strip("@").split("/")[0]
    return value or None


async def send_approved(dry_run: bool = False) -> int:
    if not settings.telegram_api_id or not settings.telegram_api_hash:
        raise RuntimeError("Задайте TELEGRAM_API_ID и TELEGRAM_API_HASH (my.telegram.org).")

    try:
        from telethon import TelegramClient
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Не установлен telethon: pip install telethon") from exc

    storage = Storage(settings.db_path)
    leads = [l for l in storage.list_leads(status=LeadStatus.APPROVED, limit=100)
             if l.best_channel is Channel.TELEGRAM]
    if not leads:
        log.info("Нет одобренных лидов с Telegram-контактом.")
        return 0

    sent = 0
    client = TelegramClient(settings.telegram_session, settings.telegram_api_id, settings.telegram_api_hash)
    async with client:
        for lead in leads:
            username = _normalize_username(lead.telegram)
            if not username:
                continue
            check_limits(storage)
            if dry_run:
                log.info("[dry-run] отправил бы @%s: %s", username, lead.message[:60].replace("\n", " "))
                sent += 1
                storage.log_send(lead.key, Channel.TELEGRAM.value)  # для проверки лимитов
                continue
            try:
                await client.send_message(username, lead.message)
            except Exception as exc:  # noqa: BLE001
                log.warning("Не удалось написать @%s: %s", username, exc)
                continue
            storage.set_status(lead.key, LeadStatus.CONTACTED)
            storage.log_send(lead.key, Channel.TELEGRAM.value)
            sent += 1
            log.info("Отправлено @%s", username)
            await asyncio.sleep(settings.min_seconds_between_sends)
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description="Отправка одобренных лидов через Telethon")
    parser.add_argument("--dry-run", action="store_true", help="Показать, что было бы отправлено")
    args = parser.parse_args()
    count = asyncio.run(send_approved(dry_run=args.dry_run))
    log.info("Итого обработано: %s", count)


if __name__ == "__main__":
    main()
