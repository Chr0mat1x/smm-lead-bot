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
    """Сценарий: владелец говорит «не тот контакт» -> агент отклоняет лид и отвечает."""
    lead = seed_lead(storage)
    llm = ScriptedLLM([
        LLMReply(tool_calls=[ToolCall(id="c1", name="reject_lead",
                                      arguments={"lead_key": lead.key, "reason": "не тот"})]),
        LLMReply(text="Убрал этот контакт."),
    ])
    agent = Agent(storage, llm)

    answer = agent.respond(chat_id=1, user_text="это не тот контакт", context_lead_key=lead.key)

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
