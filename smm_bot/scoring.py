"""Скоринг лидов: кого стоит обходить в первую очередь.

Логика простая и объяснимая: у кого выше шанс ответить и кто больше заплатит.
Скоринг не "магия", а сумма прозрачных факторов — вы всегда можете его поправить.
"""
from __future__ import annotations

from .models import Channel, Lead, LeadStatus

# Приоритет категорий: насколько бизнесу вообще нужны сайт/автоматизация.
CATEGORY_WEIGHT: dict[str, int] = {
    "cafe": 30, "restaurant": 28, "fast_food": 22, "bar": 18, "pub": 18,
    "bakery": 26, "sauna": 34, "spa": 30, "hairdresser": 26, "beauty": 30,
    "massage": 24, "fitness_centre": 26, "sports_centre": 22, "car_repair": 30,
    "tyres": 22, "car_wash": 34, "laundry": 24, "dry_cleaning": 24,
    "florist": 24, "pet_grooming": 30, "pet": 20, "doityourself": 16, "hardware": 14,
}

CHANNEL_WEIGHT: dict[Channel, int] = {
    Channel.TELEGRAM: 30,
    Channel.PHONE: 22,
    Channel.EMAIL: 16,
    Channel.INSTAGRAM: 18,
    Channel.VK: 18,
    Channel.SITE: 8,
    Channel.NONE: -40,
}

STATUS_PENALTY: dict[LeadStatus, int] = {
    LeadStatus.DO_NOT_CONTACT: -1000,
    LeadStatus.REJECTED: -500,
    LeadStatus.CONTACTED: -60,
    LeadStatus.REPLIED: 40,
}


def score_lead(lead: Lead) -> int:
    score = 0
    score += CATEGORY_WEIGHT.get(lead.category, 12)
    score += CHANNEL_WEIGHT.get(lead.best_channel, 0)
    if not lead.has_website:
        score += 25
    if lead.address:
        score += 3
    if lead.phone:
        score += 5
    score += STATUS_PENALTY.get(lead.status, 0)
    return score


def score_and_sort(leads: list[Lead]) -> list[Lead]:
    for lead in leads:
        lead.score = score_lead(lead)
    return sorted(leads, key=lambda l: l.score, reverse=True)


def filter_no_website(leads: list[Lead]) -> list[Lead]:
    """Оставляем только тех, у кого нет сайта, и к кому есть чем достучаться."""
    return [l for l in leads if not l.has_website and l.reachable]
