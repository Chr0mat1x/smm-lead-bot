"""Конфигурация проекта. Все секреты берутся из окружения (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    db_path: Path = field(default_factory=lambda: Path(os.getenv("DB_PATH", BASE_DIR / "data" / "leads.sqlite3")))

    # Telethon (полуручная отправка с вашего личного аккаунта)
    telegram_api_id: int = 0
    telegram_api_hash: str = os.getenv("TELEGRAM_API_HASH", "")
    telegram_session: str = os.getenv("TELEGRAM_SESSION", str(BASE_DIR / "data" / "userbot"))

    # Предложение, которое вы продаёте
    your_name: str = os.getenv("YOUR_NAME", "Александр")
    your_contact: str = os.getenv("YOUR_CONTACT", "@ваш_ник")
    offer_url: str = os.getenv("OFFER_URL", "")

    # Лимиты безопасности
    daily_send_limit: int = _int(os.getenv("DAILY_SEND_LIMIT"), 15)
    min_seconds_between_sends: int = _int(os.getenv("MIN_SECONDS_BETWEEN_SENDS"), 120)

    # LLM для диалога и генерации сообщений
    # по умолчанию бесплатный режим: он не требует ключа и нужен владельцу
    llm_provider: str = os.getenv("LLM_PROVIDER", "pollinations")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7") or 0.7)

    # порт для health-сервера (нужен на хостингах, которые ждут открытый порт)
    port: int = _int(os.getenv("PORT"), 0)

    allowed_user_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        ids = tuple(int(x) for x in _csv(os.getenv("ALLOWED_USER_IDS")))
        object.__setattr__(self, "allowed_user_ids", ids)
        object.__setattr__(self, "telegram_api_id", _int(os.getenv("TELEGRAM_API_ID"), 0))

    def validate_bot(self) -> None:
        if not self.telegram_bot_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN не задан. Скопируйте .env.example в .env и заполните.")


settings = Settings()
