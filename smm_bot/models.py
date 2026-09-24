"""Модели данных."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LeadStatus(str, Enum):
    NEW = "new"                 # найден, ещё не обработан
    APPROVED = "approved"       # вы одобрили, готов к отправке
    CONTACTED = "contacted"     # сообщение отправлено
    REPLIED = "replied"         # клиент ответил
    REJECTED = "rejected"       # не подходит
    DO_NOT_CONTACT = "dnc"      # попросил не писать / отписка


class Channel(str, Enum):
    SITE = "site"
    TELEGRAM = "telegram"
    PHONE = "phone"
    INSTAGRAM = "instagram"
    VK = "vk"
    EMAIL = "email"
    NONE = "none"


@dataclass
class Lead:
    source: str
    source_id: str
    name: str
    category: str = ""
    city: str = ""
    address: str = ""
    lat: float | None = None
    lon: float | None = None
    has_website: bool = False
    website: str = ""
    phone: str = ""
    email: str = ""
    instagram: str = ""
    vk: str = ""
    telegram: str = ""
    tg_kind: str = ""  # channel | group | private | invite | unknown
    tg_title: str = ""
    score: int = 0
    status: LeadStatus = LeadStatus.NEW
    message: str = ""
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.source}:{self.source_id}"

    @property
    def tg(self):
        """Telegram-контакт в приведённом виде (ссылка, тип, юзернейм)."""
        from .telegram_link import normalize  # локальный импорт против цикла

        return normalize(self.telegram, kind=self.tg_kind, title=self.tg_title)

    @property
    def has_tg_channel(self) -> bool:
        """Есть публичный канал — только такие лиды годятся для рассылки в TG.

        Группу и личный аккаунт сюда не пускаем: в группе писать нельзя без
        разрешения, а личный аккаунт в OSM чаще всего указан без согласия
        человека. Канал — единственный канал связи, где обращение уместно.
        """
        return self.tg.is_channel

    @property
    def best_channel(self) -> Channel:
        """Канал, по которому с этим лидом реально можно связаться."""
        if self.has_tg_channel:
            return Channel.TELEGRAM
        if self.phone:
            return Channel.PHONE
        if self.email:
            return Channel.EMAIL
        if self.instagram:
            return Channel.INSTAGRAM
        if self.vk:
            return Channel.VK
        if self.website:
            return Channel.SITE
        return Channel.NONE

    @property
    def reachable(self) -> bool:
        return self.best_channel is not Channel.NONE
