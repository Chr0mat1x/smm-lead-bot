"""Связующий слой: найти -> проверить Telegram -> сохранить -> отскорить."""
from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import osm, overture
from .config import settings
from .message_generator import generate_message
from .models import Lead, LeadStatus
from .scoring import filter_no_website, score_and_sort, score_lead
from .storage import Storage
from . import telegram_link

log = logging.getLogger("smm_bot.pipeline")

# сколько проверок Telegram запускаем одновременно: getChat — обычный запрос
# к Bot API, но десятки сразу вызовут лимит, поэтому держим пул небольшим
TG_LOOKUP_WORKERS = 4
# Telegram отвечает 429, если спрашивать слишком часто. Держим паузу между
# запросами: 60 проверок с ней укладываются в разумное время, но не ловят блок.
TG_LOOKUP_PAUSE = 0.6


@dataclass
class DiscoveryStats:
    found_raw: int = 0
    without_website: int = 0
    reachable: int = 0
    with_telegram: int = 0
    tg_channels: int = 0
    new: int = 0
    updated: int = 0
    overture: int = 0   # сколько объектов принёс второй источник
    city: str = ""  # каноническое название (не падеж) — по нему фильтруем выдачу

    def as_text(self) -> str:
        source_line = (f"Из них из Overture: {self.overture}\n"
                       if self.overture else "")
        return (
            f"Объектов найдено: {self.found_raw}\n"
            f"{source_line}"
            f"Без сайта: {self.without_website}\n"
            f"С контактом: {self.reachable}\n"
            f"С Telegram: {self.with_telegram} (из них каналов: {self.tg_channels})\n"
            f"Новых в базе: {self.new}\n"
            f"Обновлено: {self.updated}"
        )


def verify_telegram(leads: list[Lead], token: str | None = None,
                    max_checks: int = 60) -> int:
    """Проставить тип Telegram-контакта через getChat. Возвращает число каналов.

    Проверяем только те лиды, где Telegram вообще есть. Результат кешируется
    в базе (tg_kind), поэтому при повторном поиске запросов не будет.
    """
    token = token or settings.telegram_bot_token
    if not token:
        return 0
    pending = [l for l in leads if l.telegram and l.tg_kind in ("", "unknown")]
    if not pending:
        return sum(1 for l in leads if l.has_tg_channel)

    # раньше видели 429 у Telegram: между проверками держим паузу
    lock = threading.Lock()
    last = [0.0]

    def check(lead: Lead) -> None:
        with lock:
            wait = TG_LOOKUP_PAUSE - (time.monotonic() - last[0])
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()
        try:
            ref = telegram_link.lookup(lead.telegram, token)
        except Exception:  # noqa: BLE001 — один сбойный контакт не должен рушить поиск
            log.exception("проверка Telegram упала на @%s", lead.telegram)
            return
        lead.tg_kind = ref.kind
        lead.tg_title = ref.title or lead.tg_title

    with ThreadPoolExecutor(max_workers=TG_LOOKUP_WORKERS) as pool:
        list(pool.map(check, pending[:max_checks]))
    return sum(1 for l in leads if l.has_tg_channel)


def run_discovery(storage: Storage, place: str, categories: list[str] | None = None,
                  your_name: str | None = None,
                  verify_tg: bool = True) -> DiscoveryStats:
    """Находит лиды, чистит, проверяет Telegram, скорит, сохраняет и готовит сообщения."""
    area = osm.geocode(place)
    elements = osm.fetch(area, categories or osm.DEFAULT_CATEGORIES)
    raw_leads = osm.parse_elements(elements, city=area.name or place)
    stats = DiscoveryStats(found_raw=len(raw_leads), city=area.name or place)

    # Второй источник: Overture. OSM даёт объекты без сайта, но у российских
    # заведений там почти нет телефонов и Telegram. Overture — другой открытый
    # датасет, и он приносит контакты, которых в OSM нет. Сбой источника не
    # должен ломать поиск: OSM уже собрал лиды, поэтому ловим исключение.
    other = []
    if settings.use_overture:
        try:
            other = overture.fetch_leads(area.bbox, city=area.name or place,
                                         categories=categories)
            stats.found_raw += len(other)
            stats.overture = len(other)
        except Exception as exc:  # noqa: BLE001 — источник необязательный
            log.warning("Источник Overture недоступен, работаю на OSM: %s", exc)

    candidates = dedupe_by_brand(merge_sources(raw_leads, other))
    if verify_tg:
        stats.tg_channels = verify_telegram(candidates)
    # Оставляем тех, до кого есть чем достучаться. Лид с одним Telegram тоже
    # оставляем, даже если канал не подтвердился: телефон мог появиться позже.
    candidates = [l for l in candidates if l.reachable or l.telegram]
    candidates = score_and_sort(candidates)
    stats.without_website = len(candidates)
    stats.with_telegram = sum(1 for l in candidates if l.telegram)
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


def _norm_name(name: str) -> str:
    """Название в виде, пригодном для сравнения между источниками.

    «Кафе „Милэш“» и «Милэш» — одно и то же заведение. Приводим регистр,
    убираем кавычки, скобки, знаки и организационные слова. Если после этого
    не осталось ничего (заведение названо просто «Кофейня»), возвращаем
    очищенное название без вырезания слов: иначе все «Кофейни» схлопнулись бы
    в одно пустое имя и склеились друг с другом.
    """
    text = (name or "").lower().replace("ё", "е")
    text = re.sub(r"[«»\"'`()\[\].,!?—–-]", " ", text)
    cleaned = " ".join(text.split())
    stripped = re.sub(r"\b(ооо|ип|кафе|кофейня|ресторан|бар|салон|баня|сауна|студия|"
                      r"магазин|центр|клуб|сеть|филиал)\b", " ", cleaned)
    stripped = " ".join(stripped.split())
    return stripped or cleaned


def _same_place(a: Lead, b: Lead) -> bool:
    """Одно и то же заведение в двух источниках?

    Название — главный признак. Если названия совпали, дополнительно требуем
    совпадения города или близости координат, чтобы «Милэш» в Казани и в
    Москве не склеились в один лид.
    """
    if not _norm_name(a.name) or _norm_name(a.name) != _norm_name(b.name):
        return False
    if a.city and b.city and a.city.strip().lower() != b.city.strip().lower():
        return False
    if a.lat is not None and b.lat is not None:
        # ~2 км: заведения с одним названием в одном городе рядом — это одно место
        return abs(a.lat - b.lat) < 0.02 and abs((a.lon or 0) - (b.lon or 0)) < 0.02
    return True


def _enrich(base: Lead, extra: Lead) -> Lead:
    """Дозаполнить лид OSM данными Overture, не затирая уже найденное.

    Так одно заведение остаётся одной строкой в базе, но телефон, Instagram и
    адрес, которых в OSM нет, подтягиваются из Overture.
    """
    for field in ("phone", "email", "instagram", "vk", "telegram", "address", "website"):
        current = getattr(base, field, "")
        if not current:
            value = getattr(extra, field, "")
            if value:
                setattr(base, field, value)
    if not base.tg_kind and extra.tg_kind:
        base.tg_kind = extra.tg_kind
    if base.lat is None and extra.lat is not None:
        base.lat = extra.lat
        base.lon = extra.lon
    if not base.notes and extra.notes:
        base.notes = extra.notes
    return base


def merge_sources(primary: list[Lead], secondary: list[Lead]) -> list[Lead]:
    """Объединить лиды двух источников, не потеряв контакты.

    Правило: одно заведение — один лид. Совпавшие по названию объекты
    обогащаем контактами из второго источника, остальные добавляем как есть.
    Иначе владелец увидит одно и то же кафе дважды и напишет ему два раза.
    """
    if not secondary:
        return primary
    merged = list(primary)
    added = 0
    for lead in secondary:
        for existing in merged:
            if _same_place(existing, lead):
                if lead.source != existing.source:
                    _enrich(existing, lead)
                break
        else:
            # совпадений нет — добавляем как самостоятельный лид
            key = f"{lead.source}:{lead.source_id}"
            if not any(f"{e.source}:{e.source_id}" == key for e in merged):
                merged.append(lead)
                added += 1
    log.info("Объединение источников: OSM %s + Overture %s -> %s",
             len(primary), added, len(merged))
    return merged


def recheck_telegram(storage: Storage, city: str | None = None, limit: int = 200,
                     progress=None) -> tuple[int, int]:
    """Перепроверить тип Telegram у лидов, которые уже лежат в базе.

    Нужно, потому что до появления проверки каналов тип нигде не хранился,
    а часть старых контактов могла быть помечена «не найден» из-за лимита
    Telegram. Возвращает (сколько проверили, сколько оказались каналами).
    """
    if city:
        candidates = [l for l in storage.find_by_city(city) if l.telegram
                      and l.tg_kind in ("", "unknown", "not_found")]
    else:
        candidates = [l for l in storage._all_leads_with_tg(10000)
                      if l.tg_kind in ("", "unknown", "not_found")]
    candidates = candidates[:limit]
    if not candidates:
        return 0, 0

    # сбрасываем кеш, чтобы lookup реально сходил в Telegram
    for lead in candidates:
        lead.tg_kind = ""
    verify_telegram(candidates)
    channels = 0
    for lead in candidates:
        if lead.has_tg_channel:
            channels += 1
            # канал поднимает скор: такой лид должен идти в начале выдачи
            lead.score = max(lead.score, score_lead(lead))
        storage.set_tg_kind(lead.key, lead.tg_kind, lead.tg_title)
        storage.set_score(lead.key, lead.score)
        if progress:
            progress(lead)
    return len(candidates), channels


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
