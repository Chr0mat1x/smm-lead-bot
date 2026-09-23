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
import re

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (CallbackQuery, FSInputFile, InlineKeyboardButton,
                           InlineKeyboardMarkup, KeyboardButton, Message,
                           ReplyKeyboardMarkup)

from .agent import Agent
from .config import settings
from .health import start_health_server
from .llm import build_llm
from .models import Channel, Lead, LeadStatus
from .osm import category_tag_values
from .pipeline import export_leads_csv, prepare_messages, run_discovery
from .scoring import CATEGORY_WEIGHT
from .storage import Storage
from .tools import category_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smm_bot")

# сколько ждём ответ нейросети, прежде чем сказать «не дождался»
AGENT_TIMEOUT = 120

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

# чаты, которые сейчас ждут от нас название города
pending_city: set[int] = set()

# какие категории искали последними: «Следующие лиды» не должны подмешивать
# кафе к баням, если поиск был по баням
active_categories: dict[int, list[str]] = {}

# город последнего поиска: «Следующие лиды» должны добирать тот же город,
# а не подмешивать старые лиды из других городов
active_city: dict[int, str] = {}

# LLM и ассистент (могут быть в режиме без ключа — тогда работает запасная логика)
agent = Agent(storage, build_llm())


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
    category = category_label(lead.category)

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
        "/ask вопрос — спросить ассистента\n"
        "/reset — очистить историю диалога\n"
        "/stats — статистика базы\n"
        "/csv — выгрузить лиды в файл\n\n"
        "Просто напишите вопрос словами — ассистент ответит.\n"
        "Ответьте (reply) на карточку лида и напишите, что не так — он поправит.\n\n"
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
    # stats.city — каноническое название от геокодера: показываем «Саратов»,
    # а не «Саратове», и по нему же фильтруем выдачу
    active_city[message.chat.id] = stats.city or place
    active_categories[message.chat.id] = category_tag_values(cats)
    # Новый поиск по городу начинаем с лучших лидов, а не с того места,
    # где остановились в прошлый раз.
    storage.reset_shown_for_city(active_city[message.chat.id])
    await note.edit_text(f"Готово по «{stats.city or place}»:\n\n{stats.as_text()}")
    await show_next_leads(message, place=stats.city or place)


@dp.message(F.text == "🔍 Найти клиентов")
async def find_clients(message: Message) -> None:
    if not is_allowed(message):
        return
    pending_city.add(message.chat.id)
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
    city = active_city.get(message.chat.id)
    if not city:
        await message.answer("Сначала поиск: /find Город — потом покажу следующие лиды.")
        return
    await show_next_leads(message, place=city)


async def show_next_leads(message: Message, place: str | None = None, batch_size: int = 5) -> None:
    prepare_messages(storage)
    city = place or active_city.get(message.chat.id)
    if not city:
        await message.answer("Сначала поиск: /find Город — потом покажу лиды.")
        return

    def pick(unseen: bool) -> list[Lead]:
        return [l for l in storage.find_by_city(
                    city, status=LeadStatus.NEW, unseen_only=unseen,
                    categories=active_categories.get(message.chat.id) or None)
                if l.reachable][:batch_size]

    # Сначала только непоказанные. Если в городе новых не осталось — берём
    # уже показанные, чтобы человек мог вернуться к ним, а не получить молчание.
    leads = pick(unseen=True)
    repeat = False
    if not leads:
        leads = pick(unseen=False)
        repeat = True
    if not leads:
        await message.answer(
            f"По городу «{city}» новых лидов нет.\n"
            "Попробуйте другой город или другую категорию."
        )
        return

    current_batch[message.chat.id] = leads
    # название берём у самого лида: там каноническая форма («Саратов»), а не падеж
    shown_city = leads[0].city or city
    header = f"Город: {shown_city}. Лиды {len(leads)} шт. по приоритету."
    if repeat:
        header += "\nЭто уже показанные ранее — новых по городу не осталось."
    await message.answer(
        header + "\nПроверьте сообщение и отправьте вручную. "
        "Если контакт не тот — ответьте (reply) на карточку и напишите, что не так."
    )
    for index, lead in enumerate(leads):
        sent = await message.answer(format_lead(lead, position=f"{index + 1}. "),
                                    parse_mode="HTML", reply_markup=lead_keyboard(index))
        storage.link_message(message.chat.id, sent.message_id, lead.key)
    if not repeat:
        storage.mark_shown([l.key for l in leads])


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


@dp.message(Command("ask"))
async def ask_cmd(message: Message) -> None:
    """Явно поговорить с ассистентом."""
    if not is_allowed(message):
        return
    question = message.text.removeprefix("/ask").strip()
    await talk_to_agent(message, question or "Что сейчас в базе?")


@dp.message(Command("reset"))
async def reset_cmd(message: Message) -> None:
    """Забыть историю диалога."""
    if not is_allowed(message):
        return
    removed = storage.clear_dialogue(message.chat.id)
    agent.toolbox.last_shown = []
    await message.answer(f"Историю диалога очистил ({removed} сообщ.). Контекст начат заново.")


async def talk_to_agent(message: Message, text: str) -> None:
    """Отправить сообщение ассистенту и показать ответ."""
    context_lead_key = None
    replied = message.reply_to_message
    if replied:
        context_lead_key = storage.lead_key_for_message(message.chat.id, replied.message_id)

    note = await message.answer("Думаю…")
    chat_id = message.chat.id
    task = asyncio.create_task(
        asyncio.to_thread(agent.respond, chat_id, text, context_lead_key)
    )

    # Бесплатная модель иногда зависает. Не заставляем человека ждать вслепую:
    # каждые 15 секунд обновляем статус, а после 120 секунд отвечаем честно.
    waited = 0
    while waited < AGENT_TIMEOUT:
        try:
            answer = await asyncio.wait_for(asyncio.shield(task), timeout=15)
            break
        except asyncio.TimeoutError:
            waited += 15
            await note.edit_text(f"Думаю… уже {waited} секунд. Нейросеть отвечает медленно.")
    else:
        await note.edit_text(
            "Нейросеть не ответила за 2 минуты — видимо, сервис перегружен.\n\n"
            "Пока могу работать по командам: «найди в Саратове», «покажи следующие», "
            "«статистика». Или напишите ещё раз через минуту."
        )
        return

    await note.edit_text(render_answer(answer))


def render_answer(text: str) -> str:
    """Ответ ассистента может содержать что угодно — чистим markdown и режем длину."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # строки-разделители таблиц («|---|---|») и сами таблицы в Telegram
        # выглядят мусором: палки, звёздочки, обратные кавычки
        if re.fullmatch(r"\|?[\s|:\-]+\|?", stripped) and "|" in stripped:
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            line = " — ".join(c for c in cells if c)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", line)
        line = line.replace("`", "")
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    safe = html.escape(cleaned)
    if len(safe) > 3800:
        safe = safe[:3800] + "…"
    return safe


@dp.message(F.text)
async def free_text(message: Message) -> None:
    """Любой текст: либо название города, либо вопрос ассистенту."""
    if not is_allowed(message):
        return
    text = (message.text or "").strip()
    if not text:
        return

    if message.chat.id in pending_city and not text.endswith("?"):
        pending_city.discard(message.chat.id)
        await do_discovery(message, text)
        return

    pending_city.discard(message.chat.id)
    await talk_to_agent(message, text)


async def main() -> None:
    settings.validate_bot()
    if not settings.allowed_user_ids:
        log.warning("ALLOWED_USER_IDS не задан: бот ответит любому, кто его найдёт.")
    # на хостингах вроде Render нужен открытый порт, иначе сервис считают упавшим
    if settings.port:
        start_health_server(storage, settings.port)
    bot = Bot(token=settings.telegram_bot_token)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
