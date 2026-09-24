"""Тесты диалога с ассистентом. LLM подменяем фейком — сеть не нужна.

Фейк отдаёт заранее заданные ответы по очереди: так мы проверяем, что агент
правильно обрабатывает вызовы инструментов и реально меняет базу.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smm_bot.agent import Agent
from smm_bot.llm import BaseLLM, LLMError, LLMReply, ToolCall
from smm_bot.models import Lead, LeadStatus
from smm_bot.storage import Storage
from smm_bot.tools import ToolBox


class ScriptedLLM(BaseLLM):
    """Отдаёт заранее прописанные ответы. Запоминает, что ей передали.

    Когда инструменты не переданы (финальный вызов агента), отдаём final_text —
    так же ведёт себя живая модель, которая обязана подвести итог словами.
    """

    def __init__(self, replies: list[LLMReply], final_text: str = "") -> None:
        self.replies = list(replies)
        self.final_text = final_text
        self.seen_messages: list[list[dict]] = []
        self.seen_tools: list[list[dict] | None] = []

    def chat(self, messages, tools=None) -> LLMReply:
        self.seen_messages.append(list(messages))
        self.seen_tools.append(tools)
        if tools is None:
            return LLMReply(text=self.final_text or "(итог)")
        if not self.replies:
            return LLMReply(text="(больше ответов нет)")
        return self.replies.pop(0)


class ExplodingLLM(BaseLLM):
    def chat(self, messages, tools=None) -> LLMReply:
        raise LLMError("нет ключа")


@pytest.fixture()
def storage(tmp_path) -> Storage:
    return Storage(tmp_path / "test.sqlite3")


def seed_lead(storage: Storage, source_id: str = "node/1", name: str = "Тестовое кафе",
              category: str = "cafe", telegram: str = "@test") -> Lead:
    lead = Lead(source="osm", source_id=source_id, name=name, category=category,
                city="Тюмень", telegram=telegram, score=50)
    storage.upsert_lead(lead)
    return storage.get(lead.key)


def test_tools_schemas_include_reject_and_search(storage: Storage) -> None:
    names = {t["function"]["name"] for t in ToolBox(storage).schemas()}
    assert {"search_leads", "reject_lead", "set_message", "show_leads", "stats"} <= names


def _seed_city(storage: Storage, count: int, city: str = "Казань") -> None:
    for i in range(count):
        storage.upsert_lead(Lead(source="osm", source_id=f"n/{i}", name=f"Кафе {i}",
                                 category="cafe", city=city, phone="+79001234567",
                                 score=100 - i))


def test_show_more_advances_to_new_leads(storage: Storage) -> None:
    """«Покажи ещё лиды» должно давать новые, а не крутить первую тройку."""
    _seed_city(storage, 9)
    agent = Agent(storage, ScriptedLLM([]))
    agent.toolbox.active_city = "Казань"
    agent.toolbox.active_categories = ["cafe"]

    first = agent.quick_intent("покажи ещё лиды")
    second = agent.quick_intent("покажи ещё лиды")

    assert first and second
    assert first != second, "вторая выдача повторила первую"
    # все девять должны быть показаны ровно по разу: три итерации по три
    third = agent.quick_intent("покажи ещё лиды")
    seen = set()
    for chunk in (first, second, third):
        for line in chunk.splitlines()[1:]:
            seen.add(line.split(".")[1].strip().split(" (")[0])
    assert seen == {f"Кафе {i}" for i in range(9)}


def test_show_more_reports_when_exhausted(storage: Storage) -> None:
    """Когда непоказанных не осталось — не врём пустотой, а говорим прямо."""
    _seed_city(storage, 3)
    agent = Agent(storage, ScriptedLLM([]))
    agent.toolbox.active_city = "Казань"
    agent.toolbox.active_categories = ["cafe"]

    agent.quick_intent("покажи ещё лиды")
    answer = agent.quick_intent("покажи ещё лиды")
    assert answer is not None
    assert "не осталось" in answer


def test_show_leads_marks_only_displayed_leads_shown(storage: Storage) -> None:
    """Помечать показанными надо ровно те лиды, что ушли на экран.

    Иначе половина выдачи помечалась показанной, но не показывалась — лид
    пропадал навсегда: ни в выдаче, ни в «Следующих лидах». Тестируем именно
    показ (show_leads), а не search_leads: поиск ходит в сеть.
    """
    _seed_city(storage, 12)
    box = ToolBox(storage)
    box.active_city = "Казань"
    box.active_categories = ["cafe"]
    outcome = box.run("show_leads", {"status": "new", "limit": 5})
    assert outcome["ok"] is True

    shown = box.last_shown
    assert len(shown) == 5, "на экран уходит пять лидов — столько и помечаем"
    # помеченные не должны вернуться как непоказанные, остальные — должны
    unseen = storage.find_by_city("Казань", status=LeadStatus.NEW, unseen_only=True,
                                  categories=["cafe"], contactable_only=True)
    assert not (set(shown) & {l.key for l in unseen})
    assert len(unseen) == 7


def test_reject_lead_changes_status(storage: Storage) -> None:
    lead = seed_lead(storage)
    box = ToolBox(storage)
    outcome = box.run("reject_lead", {"lead_key": lead.key, "reason": "не тот район"})
    assert outcome["ok"] is True
    assert storage.get(lead.key).status is LeadStatus.REJECTED


def test_set_message_saves_text(storage: Storage) -> None:
    lead = seed_lead(storage)
    outcome = ToolBox(storage).run("set_message", {"lead_key": lead.key, "text": "Короткий текст"})
    assert outcome["ok"] is True
    assert storage.get(lead.key).message == "Короткий текст"


def test_update_contact_only_allowed_fields(storage: Storage) -> None:
    lead = seed_lead(storage)
    box = ToolBox(storage)
    ok = box.run("update_contact", {"lead_key": lead.key, "field": "phone", "value": "+79001234567"})
    assert ok["ok"] is True
    assert storage.get(lead.key).phone == "+79001234567"

    bad = box.run("update_contact", {"lead_key": lead.key, "field": "score", "value": "999"})
    assert bad["ok"] is False
    assert storage.get(lead.key).score == 50


def test_unknown_tool_and_missing_lead(storage: Storage) -> None:
    box = ToolBox(storage)
    assert box.run("no_such_tool", {})["ok"] is False
    assert box.run("reject_lead", {"lead_key": "osm:nope"})["ok"] is False


def test_agent_runs_tool_then_replies(storage: Storage) -> None:
    """Сценарий: свободная просьба -> агент вызывает инструмент и отвечает.

    Берём фразу, которую не перехватывает быстрый разбор команд: он нужен
    только для очевидных действий, а всё остальное делает нейросеть.
    """
    lead = seed_lead(storage)
    llm = ScriptedLLM([
        LLMReply(tool_calls=[ToolCall(id="c1", name="reject_lead",
                                      arguments={"lead_key": lead.key, "reason": "не тот"})]),
        LLMReply(text="Убрал этот контакт."),
    ])
    agent = Agent(storage, llm)

    answer = agent.respond(chat_id=1, user_text="сделай письмо для этого заведения покороче",
                           context_lead_key=lead.key)

    assert answer == "Убрал этот контакт."
    assert storage.get(lead.key).status is LeadStatus.REJECTED
    # инструменты были переданы модели
    assert llm.seen_tools[0] is not None
    # результат инструмента вернулся в модель
    last_call = llm.seen_messages[1]
    assert any(m.get("role") == "tool" and "отмечен как неподходящий" in m["content"] for m in last_call)


def test_agent_saves_dialogue(storage: Storage) -> None:
    llm = ScriptedLLM([LLMReply(text="Привет!")])
    agent = Agent(storage, llm)
    agent.respond(chat_id=7, user_text="привет")
    history = storage.get_dialogue(7)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"].startswith("привет")
    assert history[1]["content"] == "Привет!"


def test_agent_includes_reply_context(storage: Storage) -> None:
    """При reply на карточку агент должен видеть ключ лида в истории."""
    lead = seed_lead(storage)
    llm = ScriptedLLM([LLMReply(text="ок")])
    Agent(storage, llm).respond(chat_id=3, user_text="не подходит", context_lead_key=lead.key)
    user_msg = storage.get_dialogue(3)[0]["content"]
    assert lead.key in user_msg


def test_agent_falls_back_without_llm(storage: Storage) -> None:
    """Без ключа LLM бот не падает, а работает по простым командам."""
    lead = seed_lead(storage)
    agent = Agent(storage, ExplodingLLM())

    answer = agent.respond(chat_id=5, user_text="это не тот контакт", context_lead_key=lead.key)
    assert "неподходящий" in answer.lower()
    assert storage.get(lead.key).status is LeadStatus.REJECTED

    stats = agent.respond(chat_id=5, user_text="покажи статистику")
    assert "Статистика" in stats or "всего" in stats.lower()


def test_quick_intent_needs_no_llm(storage: Storage) -> None:
    """«Найди», «статистика» и «не тот контакт» выполняются без нейросети.

    Бесплатная модель часто в очереди (429) и отвечает по минуте — если эти
    команды зависят от неё, бот выглядит нерабочим, хотя нужное действие
    можно выполнить сразу.
    """
    agent = Agent(storage, ExplodingLLM())
    assert "Статистика" in agent.respond(chat_id=11, user_text="ещё раз статистику")
    assert "поиск" in agent.respond(chat_id=11, user_text="покажи следующие").lower()
    # «покажи статистику» тоже должно пониматься как статистика, а не как показ
    assert "Статистика" in agent.respond(chat_id=11, user_text="покажи статистику")
    # фраза без ясной команды уходит к модели, а без модели — честный отказ
    vague = agent.respond(chat_id=11, user_text="как думаешь, стоит ли писать кафе?")
    assert "нейросеть" in vague.lower()


def test_quick_intent_understands_category(storage: Storage) -> None:
    """«Найди кафе в Казани» должно искать кафе, а не всё подряд.

    Раньше слово «кафе» терялось, срабатывал список категорий по умолчанию,
    и в выдачу попадали салоны красоты.
    """
    agent = Agent(storage, ExplodingLLM())
    assert agent._extract_categories("найди кафе в Казани") == ["cafe"]
    assert agent._extract_categories("найди бани и сауны") == ["banya"]
    assert agent._extract_categories("найди барбершоп в Твери") == ["barber"]
    assert agent._extract_categories("найди клиентов в Твери") is None


def test_agent_respects_max_steps(storage: Storage) -> None:
    """Если модель зациклилась на инструментах, отдаём финальный текст."""
    seed_lead(storage)
    loop_replies = [LLMReply(tool_calls=[ToolCall(id=f"c{i}", name="stats", arguments={})])
                    for i in range(10)]
    llm = ScriptedLLM(loop_replies, final_text="Итог.")
    agent = Agent(storage, llm)
    assert agent.respond(chat_id=9, user_text="скажи итог") == "Итог."
    # 5 шагов с инструментами + 1 финальный вызов без инструментов
    assert len(llm.seen_tools) == 6
    assert llm.seen_tools[-1] is None


def test_fallback_extracts_city(storage: Storage) -> None:
    agent = Agent(storage, ExplodingLLM())
    assert agent._extract_city("найди клиентов в Екатеринбурге") == "Екатеринбурге"
    assert agent._extract_city("покажи Москва") == "Москва"


def test_message_link_roundtrip(storage: Storage) -> None:
    """reply на карточку должен находить нужный лид — на этом держится вся фича."""
    lead = seed_lead(storage)
    assert storage.lead_key_for_message(chat_id=1, message_id=555) is None

    storage.link_message(chat_id=1, message_id=555, lead_key=lead.key)
    assert storage.lead_key_for_message(1, 555) == lead.key
    # другой чат не должен видеть чужую связку
    assert storage.lead_key_for_message(2, 555) is None

    # перепривязка (например, сообщение отредактировано) перезаписывает связку
    storage.link_message(1, 555, "osm:node/999")
    assert storage.lead_key_for_message(1, 555) == "osm:node/999"


def test_dialogue_is_per_chat_and_clearable(storage: Storage) -> None:
    storage.add_dialogue(1, "user", "привет")
    storage.add_dialogue(2, "user", "другой чат")
    assert len(storage.get_dialogue(1)) == 1
    assert storage.get_dialogue(1)[0]["content"] == "привет"

    removed = storage.clear_dialogue(1)
    assert removed == 1
    assert storage.get_dialogue(1) == []
    assert len(storage.get_dialogue(2)) == 1  # чужую историю не трогаем


def test_dialogue_returns_oldest_first(storage: Storage) -> None:
    for i in range(5):
        storage.add_dialogue(1, "user", f"msg{i}")
    history = storage.get_dialogue(1, limit=3)
    assert [m["content"] for m in history] == ["msg2", "msg3", "msg4"]
