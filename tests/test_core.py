"""Тесты ядра: скоринг, дедупликация, генерация сообщений, парсинг OSM."""
from __future__ import annotations

import json
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


def test_poisoned_cache_is_ignored(tmp_path, monkeypatch):
    """Неполный ответ Overpass нельзя принимать за полные данные.

    Именно это ломало поиск: в кеш попали 3 объекта вместо 452, и город
    «находился» из трёх записей, пока кеш не удалишь руками.
    """
    from smm_bot import osm

    monkeypatch.setattr(osm, "CACHE_DIR", tmp_path)
    area = osm.GeoArea("Тюмень", (57.0, 65.2, 57.3, 65.8))
    cache_file = osm._cache_path(osm.build_query(area, ["cafe"]))
    cache_file.write_text(json.dumps([{"id": 1}, {"id": 2}, {"id": 3}]), encoding="utf-8")

    fresh = [{"id": i} for i in range(50)]

    class Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"elements": fresh}

    monkeypatch.setattr(osm.requests, "post", lambda *a, **k: Resp())
    assert osm.fetch(area, ["cafe"]) == fresh
    assert json.loads(cache_file.read_text(encoding="utf-8")) == fresh


def test_incomplete_overpass_answer_is_not_used(tmp_path, monkeypatch):
    """Поле remark = сервер отдал часть данных; такой ответ не принимаем."""
    from smm_bot import osm

    monkeypatch.setattr(osm, "CACHE_DIR", tmp_path)
    area = osm.GeoArea("Тюмень", (57.0, 65.2, 57.3, 65.8))

    class Timeout:
        status_code = 200

        @staticmethod
        def json():
            return {"elements": [{"id": 1}], "remark": "runtime error: Query timed out"}

    class Good:
        status_code = 200

        @staticmethod
        def json():
            return {"elements": [{"id": i} for i in range(20)]}

    answers = [Timeout(), Good()]
    monkeypatch.setattr(osm.requests, "post", lambda *a, **k: answers.pop(0))
    monkeypatch.setattr(osm.time, "sleep", lambda *_: None)
    assert len(osm.fetch(area, ["cafe"], retries=2)) == 20


def test_categories_may_arrive_as_string():
    """Нейросеть передаёт категорию строкой — раньше это роняло поиск."""
    from smm_bot import osm

    area = osm.GeoArea("Тюмень", (57.0, 65.2, 57.3, 65.8))
    assert "cafe" in osm.build_query(area, "cafe")
    # barber разворачивается в тег shop=hairdresser — важно, что категория принята
    comma = osm.build_query(area, "cafe, barber")
    assert "cafe" in comma and "hairdresser" in comma


def _lead(key: str, city: str, score: int = 50) -> Lead:
    return Lead(source="osm", source_id=key, name=f"Точка {key}", category="cafe",
                city=city, phone="+7 900 000-00-00", score=score)


def test_lead_listing_does_not_mix_cities(tmp_path) -> None:
    """Поиск в Саратове не должен показывать лиды из Санкт-Петербурга.

    Было так: «Следующие лиды» брали топ-5 по всей базе, и питерский лид
    с высоким скором вылезал после поиска в Саратове.
    """
    storage = Storage(tmp_path / "t.sqlite3")
    storage.upsert_lead(_lead("piter/1", "Санкт-Петербург", score=99))
    storage.upsert_lead(_lead("piter/2", "Санкт-Петербург", score=98))
    storage.upsert_lead(_lead("sar/1", "Саратов", score=10))

    leads = [l for l in storage.find_by_city("Саратов", status=LeadStatus.NEW) if l.reachable]
    assert [l.key for l in leads] == ["osm:sar/1"], "выдача уехала в другой город"


def test_case_form_matches_canonical_city(tmp_path) -> None:
    """В базе город хранится как «Саратов», а искать могут по «Саратове»."""
    storage = Storage(tmp_path / "t.sqlite3")
    storage.upsert_lead(_lead("sar/1", "Саратов"))
    assert storage.find_by_city("Саратове", status=LeadStatus.NEW)


def test_shown_leads_are_not_repeated(tmp_path) -> None:
    """Показанные лиды не вылезают повторно, а новый поиск сбрасывает отметки."""
    storage = Storage(tmp_path / "t.sqlite3")
    for i in range(3):
        storage.upsert_lead(_lead(f"sar/{i}", "Саратов", score=10 - i))

    first = storage.find_by_city("Саратов", status=LeadStatus.NEW, unseen_only=True)
    assert len(first) == 3
    storage.mark_shown([l.key for l in first])
    assert storage.find_by_city("Саратов", status=LeadStatus.NEW, unseen_only=True) == []

    storage.reset_shown_for_city("Саратов")
    again = storage.find_by_city("Саратов", status=LeadStatus.NEW, unseen_only=True)
    assert len(again) == 3
