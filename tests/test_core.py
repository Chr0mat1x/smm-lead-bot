"""Тесты ядра: скоринг, дедупликация, генерация сообщений, парсинг OSM."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smm_bot.message_generator import generate_message
from smm_bot.models import Channel, Lead, LeadStatus
from smm_bot.osm import parse_elements
from smm_bot.scoring import filter_no_website, score_and_sort
from smm_bot.storage import Storage


def make_lead(**kwargs) -> Lead:
    defaults = dict(source="osm", source_id="node/1", name="Тест")
    defaults.update(kwargs)
    return Lead(**defaults)


def test_best_channel_priority() -> None:
    assert make_lead(telegram="@x", phone="+7").best_channel is Channel.TELEGRAM
    assert make_lead(phone="+7").best_channel is Channel.PHONE
    assert make_lead(email="a@b.c").best_channel is Channel.EMAIL
    assert make_lead().best_channel is Channel.NONE
    assert make_lead().reachable is False


def test_filter_no_website() -> None:
    with_site = make_lead(source_id="n/1", has_website=True, phone="+7")
    no_site = make_lead(source_id="n/2", has_website=False, phone="+7")
    unreachable = make_lead(source_id="n/3", has_website=False)
    result = filter_no_website([with_site, no_site, unreachable])
    assert result == [no_site]


def test_scoring_prefers_tg_and_no_site() -> None:
    strong = make_lead(source_id="n/1", category="sauna", telegram="@x", has_website=False)
    weak = make_lead(source_id="n/2", category="cafe", phone="+7", has_website=True)
    ranked = score_and_sort([weak, strong])
    assert ranked[0] is strong
    assert ranked[0].score > ranked[1].score


def test_dnc_is_pushed_to_bottom() -> None:
    normal = make_lead(source_id="n/1", category="cafe", telegram="@x")
    dnc = make_lead(source_id="n/2", category="sauna", telegram="@x", status=LeadStatus.DO_NOT_CONTACT)
    ranked = score_and_sort([dnc, normal])
    assert ranked[0] is normal


def test_parse_elements_skips_those_with_website() -> None:
    elements = [
        {"type": "node", "id": 1, "lat": 1.0, "lon": 2.0,
         "tags": {"name": "Кафе без сайта", "amenity": "cafe", "phone": "+79000000000"}},
        {"type": "node", "id": 2, "lat": 1.1, "lon": 2.1,
         "tags": {"name": "Кафе с сайтом", "amenity": "cafe", "website": "https://x.ru"}},
        {"type": "node", "id": 3, "lat": 1.2, "lon": 2.2, "tags": {"amenity": "cafe"}},  # без имени
        {"type": "way", "id": 4, "center": {"lat": 1.3, "lon": 2.3},
         "tags": {"name": "Баня", "leisure": "sauna", "contact:telegram": "banya_tg"}},
    ]
    leads = parse_elements(elements, city="Тюмень")
    names = {l.name for l in leads}
    assert names == {"Кафе без сайта", "Баня"}
    banya = next(l for l in leads if l.name == "Баня")
    assert banya.telegram == "banya_tg"
    assert banya.lat == 1.3  # координаты взяты из center для way


def test_message_is_personalized() -> None:
    cafe = generate_message(make_lead(name="Пышка", category="cafe"), your_name="Иван")
    assert "Пышка" in cafe and "Иван" in cafe
    assert "брони" in cafe or "меню" in cafe

    sauna = generate_message(make_lead(name="Источник", category="sauna"))
    assert "запись" in sauna
    assert cafe != sauna  # разным бизнесам — разные аргументы


def test_storage_dedup(tmp_path) -> None:
    storage = Storage(tmp_path / "test.sqlite3")
    lead = make_lead(source_id="node/42", category="cafe", phone="+7")
    assert storage.upsert_lead(lead) is True     # новый
    assert storage.upsert_lead(lead) is False    # дубликат
    assert storage.total() == 1

    # статус не должен сбрасываться при повторном upsert
    storage.set_status(lead.key, LeadStatus.CONTACTED)
    storage.upsert_lead(lead)
    assert storage.get(lead.key).status is LeadStatus.CONTACTED


def test_storage_send_limits(tmp_path) -> None:
    storage = Storage(tmp_path / "test.sqlite3")
    assert storage.sent_today() == 0
    storage.log_send("osm:node/1", "telegram")
    assert storage.sent_today() == 1
    assert storage.last_send_ts() is not None
