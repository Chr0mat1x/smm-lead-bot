"""Источник Overture: разбор строк, категории, объединение с OSM и выдача лидов с телефоном."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smm_bot import overture
from smm_bot.models import Lead, LeadStatus
from smm_bot.pipeline import merge_sources
from smm_bot.storage import Storage


def row(place_id="p1", name="Кофейня", category="coffee_shop", websites=None,
        phones=None, socials=None, address="ул. Ленина 1", locality="Казань"):
    """Строка в том виде, в каком её отдаёт DuckDB."""
    return (place_id, name, category, 0.8, websites, phones, None, socials,
            address, locality, 55.79, 49.12)


def test_rows_to_leads_keeps_only_businesses_without_site() -> None:
    rows = [
        row(place_id="1", name="Без сайта", websites=None, phones=["+79001234567"]),
        row(place_id="2", name="С сайтом", websites=["https://example.com"]),
        row(place_id="3", name="Канал вместо сайта", websites=["https://t.me/cafe"]),
    ]
    leads = overture.rows_to_leads(rows, city="Казань")
    names = [l.name for l in leads]
    assert "С сайтом" not in names
    assert "Без сайта" in names
    # ссылка на Telegram в websites — это не сайт: заведение остаётся лидом
    assert "Канал вместо сайта" in names
    assert leads[0].has_website is False


def test_telegram_from_websites_becomes_handle() -> None:
    leads = overture.rows_to_leads(
        [row(place_id="1", name="Кафе", websites=["https://t.me/coffeeshop"])], city="Казань")
    assert leads[0].telegram == "coffeeshop"


def test_socials_are_parsed_into_handles() -> None:
    leads = overture.rows_to_leads(
        [row(place_id="1", name="Кафе",
             socials=["https://www.instagram.com/coffee.kzn/",
                      "https://vk.com/coffeekzn"])], city="Казань")
    assert leads[0].instagram == "coffee.kzn"
    assert leads[0].vk == "coffeekzn"


def test_category_is_stored_as_osm_value() -> None:
    """Категория пишется в OSM-значении, иначе фильтры по категориям не работают."""
    leads = overture.rows_to_leads([row(place_id="1", category="sauna")], city="Казань")
    assert leads[0].category == "sauna"
    leads = overture.rows_to_leads([row(place_id="2", category="barber")], city="Казань")
    assert leads[0].category == "hairdresser"
    leads = overture.rows_to_leads([row(place_id="3", category="spa")], city="Казань")
    assert leads[0].category == "beauty"


def test_unknown_category_is_skipped() -> None:
    assert overture.rows_to_leads([row(place_id="1", category="courthouse")], city="Казань") == []


def test_spa_category_is_disambiguated_by_name() -> None:
    """Overture помечает одним `spa` и бани, и салоны красоты."""
    def category_of(name):
        leads = overture.rows_to_leads([row(place_id="1", name=name, category="spa")], city="Казань")
        return leads[0].category

    assert category_of("Баня №10") == "sauna"
    assert category_of("Сауна Финская") == "sauna"
    assert category_of("Студия красоты LiNail") == "beauty"
    assert category_of("Romastan Barbershop") == "beauty"
    # без подсказки в названии оставляем «Салон красоты»: в spa их большинство,
    # и для выдачи «бани» лучше не показать лишний салон, чем наоборот
    assert category_of("Акватория") == "beauty"


def test_nameless_rows_are_skipped() -> None:
    assert overture.rows_to_leads([row(place_id="1", name=None)], city="Казань") == []
    assert overture.rows_to_leads([row(place_id="1", name="   ")], city="Казань") == []


def test_supported_overture_categories_accepts_preset_and_osm_names() -> None:
    by_preset = overture.supported_overture_categories(["banya"])
    by_osm_value = overture.supported_overture_categories(["sauna"])
    assert by_preset == by_osm_value
    assert "sauna" in by_preset
    # неизвестная категория не должна сужать выборку до пустого списка
    assert overture.supported_overture_categories(["нечто"]) == \
        overture.supported_overture_categories(None)


def test_build_query_filters_by_bbox_column_not_geometry() -> None:
    """Отсекаем город по столбцу bbox: предикат по geometry читает весь датасет."""
    query = overture.build_query((55.6, 48.9, 55.95, 49.4), ["cafe"])
    assert "bbox.xmin" in query and "ST_Intersects" not in query
    assert "'cafe'" in query and "'coffee_shop'" in query


def test_detect_category_field_supports_new_schema() -> None:
    """В релизе 2026-09-23 поле categories переименовано в taxonomy."""

    class FakeCon:
        def __init__(self, columns):
            self.columns = columns

        def execute(self, sql):
            assert "DESCRIBE" in sql

            class R:
                def __init__(self, rows):
                    self.rows = rows

                def fetchall(self):
                    return [[c, "VARCHAR"] for c in self.rows]

            return R(self.columns)

    assert overture.detect_category_field(FakeCon(["id", "taxonomy"]), "2026-09-23.0") == "taxonomy.primary"
    assert overture.detect_category_field(FakeCon(["id", "categories"]), "2026-08-19.0") == "categories.primary"


def test_merge_sources_enriches_osm_lead_with_overture_contacts() -> None:
    """Одно заведение — один лид, но контакты из второго источника подтягиваются."""
    osm_lead = Lead(source="osm", source_id="node/1", name="Кофейня", category="cafe",
                    city="Казань", lat=55.79, lon=49.12)
    overture_lead = Lead(source="overture", source_id="p1", name="Кофейня", category="cafe",
                         city="Казань", lat=55.79, lon=49.13, phone="+79001234567",
                         instagram="coffee")
    merged = merge_sources([osm_lead], [overture_lead])
    assert len(merged) == 1
    assert merged[0].source == "osm"
    assert merged[0].phone == "+79001234567"
    assert merged[0].instagram == "coffee"


def test_merge_sources_keeps_distinct_businesses() -> None:
    osm_lead = Lead(source="osm", source_id="node/1", name="Кофейня", city="Казань")
    other = Lead(source="overture", source_id="p1", name="Баня", city="Казань", phone="+79001234567")
    merged = merge_sources([osm_lead], [other])
    assert len(merged) == 2


def test_merge_sources_does_not_glue_same_name_in_different_cities() -> None:
    osm_lead = Lead(source="osm", source_id="node/1", name="Кофейня", city="Казань",
                    lat=55.79, lon=49.12)
    other = Lead(source="overture", source_id="p1", name="Кофейня", city="Москва",
                 lat=55.75, lon=37.62, phone="+79001234567")
    merged = merge_sources([osm_lead], [other])
    assert len(merged) == 2


def test_merge_sources_ignores_brand_words_in_names() -> None:
    osm_lead = Lead(source="osm", source_id="node/1", name="Кафе Милэш", city="Казань",
                    lat=55.79, lon=49.12)
    other = Lead(source="overture", source_id="p1", name="Милэш", city="Казань",
                 lat=55.79, lon=49.12, phone="+79001234567")
    merged = merge_sources([osm_lead], [other])
    assert len(merged) == 1
    assert merged[0].phone == "+79001234567"


def _storage_with(tmp_path, leads: list[Lead]) -> Storage:
    # Storage открывает новое соединение на каждый вызов, поэтому ":memory:"
    # не подходит: схема создаётся в одном соединении и исчезает вместе с ним
    storage = Storage(tmp_path / "leads.sqlite3")
    for lead in leads:
        storage.upsert_lead(lead)
    return storage


def test_contactable_filter_shows_phone_leads(tmp_path) -> None:
    """Главный фикс: лид с телефоном должен попадать в выдачу, а не только канал."""
    storage = _storage_with(tmp_path, [
        Lead(source="osm", source_id="1", name="С телефоном", city="Казань",
             category="cafe", phone="+79001234567", score=50),
        Lead(source="osm", source_id="2", name="С каналом", city="Казань",
             category="cafe", telegram="chan", tg_kind="channel", score=60),
        Lead(source="osm", source_id="3", name="Без контактов", city="Казань",
             category="cafe", score=70),
    ])
    shown = storage.list_leads(contactable_only=True, limit=10)
    names = {l.name for l in shown}
    assert names == {"С телефоном", "С каналом"}


def test_contactable_filter_excludes_empty_phone(tmp_path) -> None:
    storage = _storage_with(tmp_path, [
        Lead(source="osm", source_id="1", name="Пустой телефон", city="Казань", phone=""),
    ])
    assert storage.list_leads(contactable_only=True, limit=10) == []


def test_find_by_city_supports_contactable_only(tmp_path) -> None:
    storage = _storage_with(tmp_path, [
        Lead(source="osm", source_id="1", name="Кофейня", city="Казань",
             phone="+79001234567", status=LeadStatus.NEW),
    ])
    found = storage.find_by_city("Казани", status=LeadStatus.NEW, contactable_only=True)
    assert [l.name for l in found] == ["Кофейня"]
