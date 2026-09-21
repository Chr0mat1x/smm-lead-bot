"""Узнать свой Telegram user id.

Порядок: напишите вашему боту любое сообщение (например /start),
затем запустите:

    python -m scripts.get_user_id

Скрипт читает накопившиеся апдейты через getUpdates. Важно: пока работает
сам бот (long polling), этот скрипт не увидит апдейты — Telegram отдаёт
getUpdates только одному получателю. Остановите бота на время проверки.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smm_bot.config import settings  # noqa: E402


def main() -> None:
    settings.validate_bot()
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/getUpdates"
    data = requests.get(url, timeout=20).json()
    if not data.get("ok"):
        print("Telegram вернул ошибку:", data.get("description"))
        return

    seen: dict[int, str] = {}
    for update in data.get("result", []):
        message = update.get("message") or update.get("edited_message") or {}
        sender = message.get("from") or {}
        user_id = sender.get("id")
        if user_id:
            name = " ".join(x for x in [sender.get("first_name"), sender.get("last_name")] if x)
            username = f"@{sender['username']}" if sender.get("username") else "без ника"
            seen[user_id] = f"{name} ({username})"

    if not seen:
        print("Сообщений нет. Напишите боту /start и запустите скрипт снова.")
        return

    print("Найденные пользователи:")
    for user_id, label in seen.items():
        print(f"  {user_id} — {label}")
    print("\nВставьте id в .env:")
    print(f"ALLOWED_USER_IDS={','.join(str(x) for x in seen)}")


if __name__ == "__main__":
    main()
