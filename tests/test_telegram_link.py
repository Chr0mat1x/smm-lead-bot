"""Разбор Telegram-контактов, спасение ссылок из «сайта» и показ только каналов."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smm_bot.bot import format_lead, lead_keyboard
from smm_bot.llm import strip_ads
from smm_bot.models import Lead
from smm_bot.osm import parse_elements
from smm_bot.pipeline import verify_telegram
from smm_bot.storage import Storage
from smm_bot.telegram_link import extract_handle, looks_like_telegram, normalize


def test_extract_handle_from_all_shapes() -> None:
    assert extract_handle("@coffeeshop") == "coffeeshop"
    assert extract_handle("https://t.me/coffeeshop") == "coffeeshop"
    assert extract_handle("http://www.telegram.me/coffeeshop?x=1") == "coffeeshop"
    assert extract_handle("t.me/coffeeshop") == "coffeeshop"
    assert extract_handle("coffeeshop") == "coffeeshop"
    assert extract_handle("") == ""
    assert extract_handle("https://example.com") == ""
    # два имени через ; — берём первое
    assert extract_handle("coffeeshop;staff") == "coffeeshop"


def test_invite_links_are_not_usernames() -> None:
    ref = normalize("https://t.me/+AbCdEf12345")
    assert ref.is_invite
    assert ref.url == "https://t.me/+AbCdEf12345"
    assert not ref.is_channel


def test_looks_like_telegram_separates_real_sites() -> None:
    assert looks_like_telegram("https://t.me/coffeeshop")
    assert looks_like_telegram("t.me/coffeeshop")
    assert looks_like_telegram("@coffeeshop")
    assert not looks_like_telegram("https://coffeeshop.ru")
    assert not looks_like_telegram("")


def test_parse_rescues_telegram_from_website_field() -> None:
    """Бизнес без сайта, у которого в website лежит ссылка на канал, — это лид."""
    elements = [
        {"type": "node", "id": 1, "tags": {
            "name": "Кофейня", "amenity": "cafe", "website": "https://t.me/coffeeshop"}},
        {"type": "node", "id": 2, "tags": {
            "name": "С сайтом", "amenity": "cafe", "website": "https://example.ru"}},
    ]
    leads = parse_elements(elements, city="Москва")
    names = {l.name for l in leads}
    assert names == {"Кофейня"}  # настоящий сайт по-прежнему отсеян
    coffee = leads[0]
    assert coffee.has_website is False
    assert coffee.telegram == "coffeeshop"


def _fake_lookup(mapping: dict[str, str]):
    """Подменяет обращение к Bot API: возвращает заданный тип для юзернейма."""
    from smm_bot import telegram_link

    def fake(handle: str, token: str, timeout: int = 15) -> telegram_link.TelegramRef:
        kind = mapping.get(handle, "not_found")
        return telegram_link.TelegramRef(handle=handle, kind=kind)

    return fake


def test_verify_telegram_marks_channels(monkeypatch) -> None:
    from smm_bot import pipeline

    monkeypatch.setattr(pipeline.telegram_link, "lookup",
                        _fake_lookup({"coffeeshop": "channel", "barbershop": "private"}))
    leads = [
        Lead(source="osm", source_id="n/1", name="Кофейня", telegram="coffeeshop"),
        Lead(source="osm", source_id="n/2", name="Барбер", telegram="barbershop"),
        Lead(source="osm", source_id="n/3", name="Без ТГ", telegram=""),
    ]
    channels = verify_telegram(leads, token="fake")
    assert channels == 1
    assert leads[0].has_tg_channel is True
    assert leads[1].tg_kind == "private"
    assert leads[1].has_tg_channel is False
    assert leads[2].tg_kind == ""


def test_verify_telegram_without_token_does_not_crash(monkeypatch) -> None:
    from smm_bot import pipeline
    from smm_bot.config import settings

    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("getChat не должен вызываться без токена")

    monkeypatch.setattr(pipeline.telegram_link, "lookup", boom)
    # settings — frozen dataclass, меняем значение точечно
    object.__setattr__(settings, "telegram_bot_token", "")
    leads = [Lead(source="osm", source_id="n/1", name="Кофейня", telegram="coffeeshop")]
    assert verify_telegram(leads, token="") == 0
    assert called["n"] == 0


def test_recheck_revives_wrongly_marked_lead(monkeypatch, tmp_path) -> None:
    """Лид, помеченный «не найден» из-за лимита Telegram, становится каналом.

    Это и есть починка «пропавших» каналов: пока recheck не сбрасывает кеш,
    правильно найденный канал остаётся невидимым в выдаче.
    """
    from smm_bot import pipeline
    from smm_bot.config import settings

    storage = Storage(tmp_path / "leads.sqlite3")
    storage.upsert_lead(Lead(source="osm", source_id="n/1", name="Кофейня",
                             city="Москва", telegram="coffeeshop",
                             tg_kind="not_found", score=10))
    monkeypatch.setattr(pipeline.telegram_link, "lookup",
                        _fake_lookup({"coffeeshop": "channel"}))
    # recheck ходит в Telegram только когда токен задан
    object.__setattr__(settings, "telegram_bot_token", "fake")
    checked, channels = pipeline.recheck_telegram(storage, city="Москва", limit=50)
    assert (checked, channels) == (1, 1)
    saved = storage.get("osm:n/1")
    assert saved.tg_kind == "channel"
    # канал поднял скор и теперь виден в выдаче «только каналы»
    assert saved.score > 10
    assert [l.name for l in storage.find_by_city("Москва", tg_channel_only=True)] == ["Кофейня"]


def test_storage_filters_to_channels_only(tmp_path) -> None:
    storage = Storage(tmp_path / "leads.sqlite3")
    storage.upsert_lead(Lead(source="osm", source_id="n/1", name="Канал",
                             city="Москва", telegram="coffeeshop", tg_kind="channel", score=10))
    storage.upsert_lead(Lead(source="osm", source_id="n/2", name="Человек",
                             city="Москва", telegram="barbershop", tg_kind="private", score=99))
    storage.upsert_lead(Lead(source="osm", source_id="n/3", name="Без ТГ",
                             city="Москва", phone="+7", score=50))

    only_channels = storage.list_leads(city="Москва", tg_channel_only=True)
    assert [l.name for l in only_channels] == ["Канал"]

    everything = storage.find_by_city("Москва", tg_channel_only=False)
    assert len(everything) == 3
    # фильтр каналов работает и в find_by_city (им пользуется выдача бота)
    assert [l.name for l in storage.find_by_city("Москва", tg_channel_only=True)] == ["Канал"]


def test_card_shows_clickable_channel_link() -> None:
    lead = Lead(source="osm", source_id="n/1", name="Кофейня", city="Москва",
                telegram="coffeeshop", tg_kind="channel", tg_title="Кофейня",
                message="Здравствуйте")
    card = format_lead(lead)
    assert "https://t.me/coffeeshop" in card
    assert "канал" in card

    keyboard = lead_keyboard(0, lead)
    buttons = [b for row in keyboard.inline_keyboard for b in row]
    assert any(b.url == "https://t.me/coffeeshop" for b in buttons)
    # у лида без Telegram кнопки-ссылки быть не должно
    assert all(b.url is None for row in lead_keyboard(0).inline_keyboard for b in row)


def test_strip_ads_removes_pollinations_banner() -> None:
    raw = ("1. Кофейня — кафе, Москва\n\n---\n\n**Support Pollinations.AI:**\n\n"
           "🌸 **Ad** 🌸\nPowered by Pollinations.AI")
    assert strip_ads(raw) == "1. Кофейня — кафе, Москва"
    # обычный текст без рекламы не портим
    assert strip_ads("Просто ответ.") == "Просто ответ."


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = b"{}"

    def json(self) -> dict:
        return self._payload


def test_lookup_treats_rate_limit_as_unknown(monkeypatch) -> None:
    """429 от Telegram — это «проверим позже», а не «канала нет».

    Раньше лимит сохранялся как not_found, и настоящие каналы молча пропадали
    из выдачи: именно так Санкт-Петербург показал ноль каналов.
    """
    from smm_bot import telegram_link

    monkeypatch.setattr(telegram_link.requests, "get",
                        lambda *a, **k: _FakeResponse(429, {"ok": False,
                                                            "description": "Too Many Requests"}))
    ref = telegram_link.lookup("coffeeshop", token="fake")
    assert ref.kind == "unknown"


def test_lookup_rate_limit_inside_200_is_unknown(monkeypatch) -> None:
    from smm_bot import telegram_link

    monkeypatch.setattr(telegram_link.requests, "get",
                        lambda *a, **k: _FakeResponse(200, {"ok": False,
                                                            "description": "Too Many Requests: retry after 30"}))
    assert telegram_link.lookup("coffeeshop", token="fake").kind == "unknown"


def test_lookup_not_found_stays_not_found(monkeypatch) -> None:
    from smm_bot import telegram_link

    monkeypatch.setattr(telegram_link.requests, "get",
                        lambda *a, **k: _FakeResponse(200, {"ok": False,
                                                            "description": "Bad Request: chat not found"}))
    assert telegram_link.lookup("coffeeshop", token="fake").kind == "not_found"


def test_lookup_network_error_is_unknown(monkeypatch) -> None:
    from smm_bot import telegram_link

    def boom(*a, **k):
        raise telegram_link.requests.ConnectionError("обрыв")

    monkeypatch.setattr(telegram_link.requests, "get", boom)
    assert telegram_link.lookup("coffeeshop", token="fake").kind == "unknown"
