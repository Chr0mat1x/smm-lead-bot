"""Хранилище лидов на SQLite. Дедупликация по (source, source_id)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

from .models import Lead, LeadStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    key          TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    name         TEXT NOT NULL,
    category     TEXT DEFAULT '',
    city         TEXT DEFAULT '',
    address      TEXT DEFAULT '',
    lat          REAL,
    lon          REAL,
    has_website  INTEGER DEFAULT 0,
    website      TEXT DEFAULT '',
    phone        TEXT DEFAULT '',
    email        TEXT DEFAULT '',
    instagram    TEXT DEFAULT '',
    vk           TEXT DEFAULT '',
    telegram     TEXT DEFAULT '',
    tg_kind      TEXT DEFAULT '',
    tg_title     TEXT DEFAULT '',
    score        INTEGER DEFAULT 0,
    status       TEXT DEFAULT 'new',
    message      TEXT DEFAULT '',
    notes        TEXT DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status);
CREATE INDEX IF NOT EXISTS idx_leads_score ON leads(score);

CREATE TABLE IF NOT EXISTS send_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_key   TEXT NOT NULL,
    channel    TEXT NOT NULL,
    sent_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_send_log_date ON send_log(sent_at);

CREATE TABLE IF NOT EXISTS dialogue (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    lead_key   TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dialogue_chat ON dialogue(chat_id, id);

-- связка "сообщение бота в чате" -> лид, чтобы обрабатывать reply
CREATE TABLE IF NOT EXISTS message_links (
    chat_id    INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    lead_key   TEXT NOT NULL,
    PRIMARY KEY (chat_id, message_id)
);
"""


class Storage:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Догоняем схему старых баз: CREATE TABLE IF NOT EXISTS не добавляет колонки."""
        columns = {row[1] for row in conn.execute("PRAGMA table_info(leads)")}
        if "shown_at" not in columns:
            conn.execute("ALTER TABLE leads ADD COLUMN shown_at TEXT")
        # тип Telegram-контакта (канал/группа/личный) появился позже:
        # без него нельзя понять, годится ли лид для обращения в Telegram
        if "tg_kind" not in columns:
            conn.execute("ALTER TABLE leads ADD COLUMN tg_kind TEXT DEFAULT ''")
        if "tg_title" not in columns:
            conn.execute("ALTER TABLE leads ADD COLUMN tg_title TEXT DEFAULT ''")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------- лиды ----------

    def upsert_lead(self, lead: Lead) -> bool:
        """Возвращает True, если лид новый (или обновил контакты)."""
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
        with self._conn() as conn:
            row = conn.execute("SELECT key, status FROM leads WHERE key = ?", (lead.key,)).fetchone()
            if row is None:
                conn.execute(
                    """INSERT INTO leads (key, source, source_id, name, category, city, address, lat, lon,
                        has_website, website, phone, email, instagram, vk, telegram, tg_kind, tg_title,
                        score, status, message, notes, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (lead.key, lead.source, lead.source_id, lead.name, lead.category, lead.city,
                     lead.address, lead.lat, lead.lon, int(lead.has_website), lead.website, lead.phone,
                     lead.email, lead.instagram, lead.vk, lead.telegram, lead.tg_kind, lead.tg_title,
                     lead.score, lead.status.value, lead.message, lead.notes, now, now),
                )
                return True

            # не трогаем статус, но дозаполняем контакты
            conn.execute(
                """UPDATE leads SET
                     name=COALESCE(NULLIF(?, ''), name),
                     category=COALESCE(NULLIF(?, ''), category),
                     city=COALESCE(NULLIF(?, ''), city),
                     address=COALESCE(NULLIF(?, ''), address),
                     phone=COALESCE(NULLIF(?, ''), phone),
                     email=COALESCE(NULLIF(?, ''), email),
                     instagram=COALESCE(NULLIF(?, ''), instagram),
                     vk=COALESCE(NULLIF(?, ''), vk),
                     telegram=COALESCE(NULLIF(?, ''), telegram),
                     tg_kind=COALESCE(NULLIF(?, ''), tg_kind),
                     tg_title=COALESCE(NULLIF(?, ''), tg_title),
                     website=COALESCE(NULLIF(?, ''), website),
                     has_website=MAX(has_website, ?),
                     score=MAX(score, ?),
                     updated_at=?
                   WHERE key=?""",
                (lead.name, lead.category, lead.city, lead.address, lead.phone, lead.email,
                 lead.instagram, lead.vk, lead.telegram, lead.tg_kind, lead.tg_title, lead.website,
                 int(lead.has_website), lead.score, now, lead.key),
            )
            return False

    def get(self, key: str) -> Lead | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM leads WHERE key = ?", (key,)).fetchone()
        return _row_to_lead(row) if row else None

    def list_leads(self, status: LeadStatus | None = None, limit: int = 10,
                   order: str = "score DESC, created_at DESC",
                   city: str | None = None, unseen_only: bool = False,
                   categories: list[str] | None = None,
                   tg_channel_only: bool = False,
                   contactable_only: bool = False) -> list[Lead]:
        """Список лидов с фильтрами.

        Фильтр city обязателен при показе после поиска: без него всплывают
        старые лиды других городов с более высоким скором, и выглядит это
        так, будто бот искал в Санкт-Петербурге вместо Саратова. По той же
        причине нужен фильтр categories — иначе в выдачу «бани» попадали кафе.

        tg_channel_only оставляет только публичные Telegram-каналы.

        contactable_only оставляет тех, кому вообще можно написать: канал или
        телефон. Это важно, потому что в OSM у российских заведений Telegram
        почти не указан, а телефон есть у большинства. Раньше выдача требовала
        канал и была пустой, хотя в базе лежали сотни лидов с телефонами.
        """
        query = "SELECT * FROM leads"
        where: list[str] = []
        params: list = []
        if status:
            where.append("status = ?")
            params.append(status.value)
        if city:
            where.append("city LIKE ?")
            params.append(f"%{city.strip()}%")
        if categories:
            from .osm import category_tag_values  # локальный импорт против цикла

            values = category_tag_values(categories)
            if values:
                placeholders = ", ".join("?" for _ in values)
                where.append(f"category IN ({placeholders})")
                params.extend(values)
        if unseen_only:
            where.append("(shown_at IS NULL OR shown_at = '')")
        if tg_channel_only:
            where.append("tg_kind = 'channel'")
        if contactable_only:
            where.append("(tg_kind = 'channel' OR (phone IS NOT NULL AND phone != ''))")
        if where:
            query += " WHERE " + " AND ".join(where)
        query += f" ORDER BY {order} LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_lead(r) for r in rows]

    def find_by_city(self, city: str, status: LeadStatus | None = None,
                     unseen_only: bool = False,
                     categories: list[str] | str | None = None,
                     tg_channel_only: bool = False,
                     contactable_only: bool = False) -> list[Lead]:
        """Все лиды города — чтобы «Следующие лиды» не уезжали в другой город.

        Город в базе хранится в именительном падеже («Саратов»), а приходит
        из чата падеж («Саратове») — поэтому перебираем варианты написания.
        """
        from .osm import name_variants  # локальный импорт против цикла

        for name in name_variants(city):
            found = self.list_leads(status=status, limit=10000, city=name,
                                    unseen_only=unseen_only, categories=categories,
                                    tg_channel_only=tg_channel_only,
                                    contactable_only=contactable_only)
            if found:
                return found
        return []

    def reset_shown_for_city(self, city: str) -> None:
        """Новый поиск по городу — показываем лучших заново, с первого лида."""
        if not city:
            return
        with self._conn() as conn:
            conn.execute("UPDATE leads SET shown_at = NULL WHERE city LIKE ?",
                         (f"%{city.strip()}%",))

    def mark_shown(self, keys: list[str]) -> None:
        """Помечаем показанные лиды, чтобы при следующем нажатии шли новые."""
        if not keys:
            return
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
        with self._conn() as conn:
            conn.executemany("UPDATE leads SET shown_at = ? WHERE key = ?",
                             [(now, key) for key in keys])

    def set_status(self, key: str, status: LeadStatus) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE leads SET status=?, updated_at=? WHERE key=?",
                         (status.value, datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"), key))

    def set_message(self, key: str, message: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE leads SET message=?, updated_at=? WHERE key=?",
                         (message, datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"), key))

    def set_score(self, key: str, score: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE leads SET score=?, updated_at=? WHERE key=?",
                         (score, datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"), key))

    def set_tg_kind(self, key: str, kind: str, title: str = "") -> None:
        """Записать проверенный тип Telegram-контакта (кеш для будущих поисков)."""
        with self._conn() as conn:
            conn.execute("UPDATE leads SET tg_kind=?, tg_title=?, updated_at=? WHERE key=?",
                         (kind, title, datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"), key))

    def unclassified_tg(self, limit: int = 500) -> list[Lead]:
        """Лиды, у которых Telegram есть, но тип ещё не подтверждён.

        Сюда попадают и «unknown» после 429 — их стоит перепроверить, когда
        лимит Telegram спадёт: раньше такие контакты навсегда гасились как
        «не найден», и настоящие каналы терялись.
        """
        return [l for l in self._all_leads_with_tg(limit)
                if l.tg_kind in ("", "unknown")]

    def _all_leads_with_tg(self, limit: int) -> list[Lead]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM leads WHERE telegram != '' LIMIT ?", (limit,)).fetchall()
        return [_row_to_lead(r) for r in rows]

    def count_tg_channels(self) -> int:
        """Сколько лидов имеют подтверждённый публичный Telegram-канал."""
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM leads WHERE tg_kind = 'channel'").fetchone()[0]

    def counts_by_status(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS c FROM leads GROUP BY status").fetchall()
        return {r["status"]: r["c"] for r in rows}

    def total(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]

    # ---------- журнал отправок ----------

    def log_send(self, lead_key: str, channel: str) -> None:
        with self._conn() as conn:
            conn.execute("INSERT INTO send_log (lead_key, channel, sent_at) VALUES (?,?,?)",
                         (lead_key, channel, datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")))

    def sent_today(self) -> int:
        today = date.today().isoformat()
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM send_log WHERE sent_at LIKE ?", (f"{today}%",)
            ).fetchone()[0]

    def last_send_ts(self) -> datetime | None:
        with self._conn() as conn:
            row = conn.execute("SELECT sent_at FROM send_log ORDER BY id DESC LIMIT 1").fetchone()
        return datetime.fromisoformat(row["sent_at"]) if row else None

    # ---------- диалог с ассистентом ----------

    def add_dialogue(self, chat_id: int, role: str, content: str, lead_key: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO dialogue (chat_id, role, content, lead_key, created_at) VALUES (?,?,?,?,?)",
                (chat_id, role, content, lead_key,
                 datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")),
            )

    def get_dialogue(self, chat_id: int, limit: int = 20) -> list[dict[str, str]]:
        """Последние сообщения в прямом порядке: старые -> новые."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role, content FROM dialogue WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def clear_dialogue(self, chat_id: int) -> int:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM dialogue WHERE chat_id = ?", (chat_id,))
        return cur.rowcount

    # ---------- привязка сообщений к лидам ----------

    def link_message(self, chat_id: int, message_id: int, lead_key: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO message_links (chat_id, message_id, lead_key) VALUES (?,?,?)",
                (chat_id, message_id, lead_key),
            )

    def lead_key_for_message(self, chat_id: int, message_id: int) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT lead_key FROM message_links WHERE chat_id = ? AND message_id = ?",
                (chat_id, message_id),
            ).fetchone()
        return row["lead_key"] if row else None


def _row_to_lead(row: sqlite3.Row) -> Lead:
    return Lead(
        source=row["source"], source_id=row["source_id"], name=row["name"],
        category=row["category"], city=row["city"], address=row["address"],
        lat=row["lat"], lon=row["lon"], has_website=bool(row["has_website"]),
        website=row["website"], phone=row["phone"], email=row["email"],
        instagram=row["instagram"], vk=row["vk"], telegram=row["telegram"],
        tg_kind=row["tg_kind"] or "", tg_title=row["tg_title"] or "",
        score=row["score"], status=LeadStatus(row["status"]), message=row["message"],
        notes=row["notes"],
    )
