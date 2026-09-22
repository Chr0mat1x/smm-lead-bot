"""Связующий слой: найти -> сохранить -> отскорить -> сгенерировать сообщения."""
from __future__ import annotations

from dataclasses import dataclass

from . import osm
from .config import settings
from .message_generator import generate_message
from .models import Lead, LeadStatus
from .scoring import filter_no_website, score_and_sort
from .storage import Storage


@dataclass
class DiscoveryStats:
    found_raw: int = 0
    without_website: int = 0
    reachable: int = 0
    new: int = 0
    updated: int = 0
    city: str = ""  # каноническое название (не падеж) — по нему фильтруем выдачу

    def as_text(self) -> str:
        return (
            f"Объектов найдено: {self.found_raw}\n"
            f"Без сайта: {self.without_website}\n"
            f"С контактом: {self.reachable}\n"
            f"Новых в базе: {self.new}\n"
            f"Обновлено: {self.updated}"
        )


def run_discovery(storage: Storage, place: str, categories: list[str] | None = None,
                  your_name: str | None = None) -> DiscoveryStats:
    """Находит лиды, чистит, скорит, сохраняет и готовит сообщения."""
    area = osm.geocode(place)
    elements = osm.fetch(area, categories or osm.DEFAULT_CATEGORIES)
    raw_leads = osm.parse_elements(elements, city=area.name or place)
    stats = DiscoveryStats(found_raw=len(raw_leads), city=area.name or place)

    candidates = filter_no_website(raw_leads)
    stats.without_website = len(candidates)
    candidates = score_and_sort(candidates)
    candidates = dedupe_by_brand(candidates)
    stats.reachable = sum(1 for l in candidates if l.reachable)

    display_name = your_name or settings.your_name
    for lead in candidates:
        # сообщение генерируем сразу, чтобы вы могли просмотреть его в боте
        lead.message = generate_message(lead, your_name=display_name)
        if storage.upsert_lead(lead):
            stats.new += 1
        else:
            stats.updated += 1

    return stats


def dedupe_by_brand(leads: list[Lead]) -> list[Lead]:
    """Fold branches of one chain into a single lead.

    OSM stores every address as a separate object, so a chain like "Maksim"
    can appear five times. For outreach we want one contact per brand,
    otherwise the owner gets the same message several times.
    """
    best: dict[tuple[str, str], Lead] = {}
    for lead in leads:
        key = (lead.name.strip().lower(), lead.city.strip().lower())
        current = best.get(key)
        if current is None or lead.score > current.score:
            best[key] = lead
    return list(best.values())


def prepare_messages(storage: Storage, limit: int = 10) -> int:
    """Догенерировать сообщения для лидов без текста (например, после обновления)."""
    count = 0
    for lead in storage.list_leads(status=LeadStatus.NEW, limit=limit):
        if not lead.message:
            storage.set_message(lead.key, generate_message(lead, your_name=settings.your_name))
            count += 1
    return count


def export_leads_csv(storage: Storage, path: str, status: LeadStatus | None = None,
                     limit: int = 500) -> str:
    """Выгрузка в CSV — удобно для ручного обхода и для CRM."""
    import csv

    leads = storage.list_leads(status=status, limit=limit)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["Название", "Категория", "Город", "Адрес", "Телефон", "Telegram",
                         "Instagram", "VK", "Email", "Канал", "Скор", "Статус", "Сообщение"])
        for l in leads:
            writer.writerow([l.name, l.category, l.city, l.address, l.phone, l.telegram,
                             l.instagram, l.vk, l.email, l.best_channel.value, l.score,
                             l.status.value, l.message])
    return path
