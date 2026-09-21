"""Telegram-бот для управления лидами.

Запуск:  python -m smm_bot.bot

Бот не рассылает ничего сам. Он показывает лиды, готовые сообщения и
кнопки "одобрить / отклонить / отправил / не писать". Отправку вы
подтверждаете вручную — так вы не теряете аккаунт и не спамите наугад.
"""
from __future__ import annotations

import asyncio
import html
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (CallbackQuery, FSInputFile, InlineKeyboardButton,
                           InlineKeyboardMarkup, KeyboardButton, Message,
                           ReplyKeyboardMarkup)

from .config import settings
from .models import Channel, Lead, LeadStatus
from .pipeline import export_leads_csv, prepare_messages, run_discovery
from .scoring import CATEGORY_WEIGHT
from .storage import Storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smm_bot")

storage = Storage(settings.db_path)
dp = Dispatcher()

CATEGORY_LABELS = {
    "cafe": "Кафе", "restaurant": "Рестораны", "fast_food": "Фастфуд", "bar": "Бары",
    "bakery": "Пекарни", "banya": "Бани и сауны", "barber": "Парикмахерские",
    "beauty": "Салоны красоты", "gym": "Фитнес", "car_service": "Автосервисы",
    "laundry": "Химчистки", "florist": "Цветочные", "pet": "Зооуслуги",
    "auto_wash": "Автомойки", "diy": "Строймагазины",
}
DEFAULT_CATEGORIES_UI = ["cafe", "banya", "barber", "beauty", "bakery", "auto_wash", "car_service"]

# Выбранные пользователем категории (по chat_id). В памяти — этого достаточно для одного владельца.
selected_categories: dict[int, list[str]] = {}


def is_allowed(message_or_call) -> bool:
    if not settings.allowed_user_ids:
        return True  # список не задан — бот открыт только тому, у кого есть токен
    user = getattr(message_or_call, "from_user", None)
    return bool(user and user.id in settings.allowed_user_ids)


def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔍 Найти клиентов")],
            [KeyboardButton(text="📋 Следующие лиды"), KeyboardButton(text="📊 Статистика")],
            [KeyboardButton(text="🗂 Категории"), KeyboardButton(text="📤 Экспорт CSV")],
        ],
        resize_keyboard=True,
    )


def categories_menu(chat_id: int) -> InlineKeyboardMarkup:
    chosen = set(selected_categories.get(chat_id, DEFAULT_CATEGORIES_UI))
    rows = []
    for key, label in CATEGORY_LABELS.items():
        mark = "✅ " if key in chosen else "⬜️ "
        rows.append([InlineKeyboardButton(text=f"{mark}{label}", callback_data=f"cat:{key}")])
    rows.append([InlineKeyboardButton(text="Готово", callback_data="cat:done")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def lead_keyboard(index: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="✅ Одобрить", callback_data=f"lead:approve:{index}"),
        InlineKeyboardButton(text="❌ Не подходит", callback_data=f"lead:reject:{index}"),
    ], [
        InlineKeyboardButton(text="📨 Отправил вручную", callback_data=f"lead:sent:{index}"),
        InlineKeyboardButton(text="🚫 Не писать", callback_data=f"lead:dnc:{index}"),
    ]])


# Кэш текущей выдачи, чтобы кнопки ссылались на конкретные лиды.
current_batch: dict[int, list[Lead]] = {}


def format_lead(lead: Lead, position: str = "") -> str:
    contact_lines = []
    if lead.phone:
        contact_lines.append(f"📞 {lead.phone}")
    if lead.telegram:
        contact_lines.append(f"Telegram: {lead.telegram}")
    if lead.instagram:
        contact_lines.append(f"Instagram: {lead.instagram}")
    if lead.vk:
        contact_lines.append(f"VK: {lead.vk}")
    if lead.email:
        contact_lines.append(f"✉️ {lead.email}")
    if not contact_lines:
        contact_lines.append("⚠️ Контактов нет — только адрес, нужен ручной поиск")

    location = ", ".join(x for x in [lead.city, lead.address] if x) or "адрес не указан"
    category = CATEGORY_LABELS.get(lead.category, lead.category or "—")

    esc = html.escape
    contacts = esc("\n".join(contact_lines))
    body = esc(lead.message)
    return (
        f"{position}<b>{esc(lead.name)}</b>\n"
        f"{esc(category)} · {esc(location)}\n"
        f"Скор: {lead.score} · канал: {lead.best_channel.value}\n"
        f"{contacts}\n\n"
        f"💬 <i>Готовое сообщение:</i>\n{body}"
    )


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    user = message.from_user
    if user:
        log.info("Кто-то запустил бота: id=%s username=@%s name=%s",
                 user.id, user.username, user.full_name)
    if not is_allowed(message):
        await message.answer("Бот приватный.")
        return
    await message.answer(
        "Привет! Я ищу малые бизнесы без сайта и готовлю персональные сообщения.\n\n"
        "Важно: я ничего не рассылаю сам. Вы одобряете лид и отправляете сообщение "
        "вручную — так ваш аккаунт остаётся в безопасности, а люди не жалуются на спам.\n\n"
        "Начните с «🔍 Найти клиентов».",
        reply_markup=main_menu(),
    )


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "/start — меню\n"
        "/find Москва — найти клиентов в городе\n"
        "/stats — статистика базы\n"
        "/csv — выгрузить лиды в файл\n\n"
        "❗️ Перед отправкой проверьте: у человека должно быть согласие на получение "
        "сообщений, либо вы пишете не рекламу, а личное деловое предложение с возможностью отказа.",
        reply_markup=main_menu(),
    )


@dp.message(F.text == "🗂 Категории")
async def choose_categories(message: Message) -> None:
    if not is_allowed(message):
        return
    await message.answer("Что искать? Отметьте категории:", reply_markup=categories_menu(message.chat.id))


@dp.callback_query(F.data.startswith("cat:"))
async def toggle_category(call: CallbackQuery) -> None:
    value = call.data.split(":", 1)[1]
    chat_id = call.message.chat.id
    chosen = selected_categories.setdefault(chat_id, list(DEFAULT_CATEGORIES_UI))
    if value == "done":
        labels = [CATEGORY_LABELS[c] for c in chosen]
        await call.message.edit_text("Выбрано: " + ", ".join(labels))
        await call.answer("Сохранено")
        return
    if value in chosen:
        chosen.remove(value)
    else:
        chosen.append(value)
    await call.message.edit_reply_markup(reply_markup=categories_menu(chat_id))
    await call.answer()


async def do_discovery(message: Message, place: str) -> None:
    cats = selected_categories.get(message.chat.id) or DEFAULT_CATEGORIES_UI
    note = await message.answer(f"Ищу в «{place}». Это займёт до минуты…")
    try:
        stats = await asyncio.to_thread(run_discovery, storage, place, cats)
    except Exception as exc:  # noqa: BLE001 — показываем причину пользователю
        log.exception("discovery failed")
        await note.edit_text(f"Не получилось: {exc}\n\nПопробуйте другое название города.")
        return
    await note.edit_text(f"Готово по «{place}»:\n\n{stats.as_text()}")
    await show_next_leads(message, place=place)


@dp.message(F.text == "🔍 Найти клиентов")
async def find_clients(message: Message) -> None:
    if not is_allowed(message):
        return
    await message.answer("Напишите город, например: Тюмень или Тюмень, Центральный район")


@dp.message(Command("find"))
async def find_cmd(message: Message) -> None:
    if not is_allowed(message):
        return
    place = message.text.removeprefix("/find").strip()
    if not place:
        await message.answer("Напишите город: /find Тюмень")
        return
    await do_discovery(message, place)


@dp.message(F.text == "📋 Следующие лиды")
async def next_leads_button(message: Message) -> None:
    await show_next_leads(message)


async def show_next_leads(message: Message, place: str | None = None, batch_size: int = 5) -> None:
    prepare_messages(storage)
    leads = [l for l in storage.list_leads(status=LeadStatus.NEW, limit=batch_size) if l.reachable]
    if not leads:
        await message.answer("Новых лидов нет. Запустите поиск: /find Город")
        return

    current_batch[message.chat.id] = leads
    await message.answer(
        f"Показываю {len(leads)} лид(ов) с наивысшим приоритетом. "
        "Проверьте сообщение и либо отправьте вручную, либо отметьте статус."
    )
    for index, lead in enumerate(leads):
        await message.answer(format_lead(lead, position=f"{index + 1}. "),
                             parse_mode="HTML", reply_markup=lead_keyboard(index))


@dp.callback_query(F.data.startswith("lead:"))
async def handle_lead_action(call: CallbackQuery) -> None:
    _, action, index_raw = call.data.split(":")
    index = int(index_raw)
    batch = current_batch.get(call.message.chat.id, [])
    if index >= len(batch):
        await call.answer("Лид устарел, запросите список заново", show_alert=True)
        return
    lead = batch[index]

    mapping = {
        "approve": (LeadStatus.APPROVED, "Одобрено. Отправьте сообщение вручную."),
        "reject": (LeadStatus.REJECTED, "Отмечено как неподходящее."),
        "sent": (LeadStatus.CONTACTED, "Записал: вы отправили сообщение."),
        "dnc": (LeadStatus.DO_NOT_CONTACT, "Записал: этому больше не писать."),
    }
    status, text = mapping[action]
    storage.set_status(lead.key, status)
    if action == "sent":
        storage.log_send(lead.key, lead.best_channel.value)
    await call.answer(text, show_alert=False)
    await call.message.edit_reply_markup(reply_markup=None)


@dp.message(F.text == "📊 Статистика")
async def stats_button(message: Message) -> None:
    if not is_allowed(message):
        return
    counts = storage.counts_by_status()
    lines = [f"Всего лидов: {storage.total()}", f"Отправлено сегодня: {storage.sent_today()}"]
    for status in LeadStatus:
        lines.append(f"{status.value}: {counts.get(status.value, 0)}")
    await message.answer("\n".join(lines))


@dp.message(F.text == "📤 Экспорт CSV")
async def export_button(message: Message) -> None:
    if not is_allowed(message):
        return
    path = settings.db_path.parent / "leads_export.csv"
    export_leads_csv(storage, str(path))
    await message.answer_document(FSInputFile(path), caption=f"Выгрузка лидов, всего в базе: {storage.total()}")


@dp.message(F.text.regexp(r"^[\w\s\-,\.]{2,60}$"))
async def city_input(message: Message) -> None:
    """Свободный текст трактуем как название города."""
    if not is_allowed(message):
        return
    await do_discovery(message, message.text.strip())


async def main() -> None:
    settings.validate_bot()
    if not settings.allowed_user_ids:
        log.warning("ALLOWED_USER_IDS не задан: бот ответит любому, кто его найдёт.")
    bot = Bot(token=settings.telegram_bot_token)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
