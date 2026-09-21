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
"""


class Storage:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

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
                        has_website, website, phone, email, instagram, vk, telegram, score, status, message,
                        notes, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (lead.key, lead.source, lead.source_id, lead.name, lead.category, lead.city,
                     lead.address, lead.lat, lead.lon, int(lead.has_website), lead.website, lead.phone,
                     lead.email, lead.instagram, lead.vk, lead.telegram, lead.score, lead.status.value,
                     lead.message, lead.notes, now, now),
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
                     website=COALESCE(NULLIF(?, ''), website),
                     has_website=MAX(has_website, ?),
                     score=MAX(score, ?),
                     updated_at=?
                   WHERE key=?""",
                (lead.name, lead.category, lead.city, lead.address, lead.phone, lead.email,
                 lead.instagram, lead.vk, lead.telegram, lead.website, int(lead.has_website),
                 lead.score, now, lead.key),
            )
            return False

    def get(self, key: str) -> Lead | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM leads WHERE key = ?", (key,)).fetchone()
        return _row_to_lead(row) if row else None

    def list_leads(self, status: LeadStatus | None = None, limit: int = 10,
                   order: str = "score DESC, created_at DESC") -> list[Lead]:
        query = "SELECT * FROM leads"
        params: list = []
        if status:
            query += " WHERE status = ?"
            params.append(status.value)
        query += f" ORDER BY {order} LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_lead(r) for r in rows]

    def set_status(self, key: str, status: LeadStatus) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE leads SET status=?, updated_at=? WHERE key=?",
                         (status.value, datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"), key))

    def set_message(self, key: str, message: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE leads SET message=?, updated_at=? WHERE key=?",
                         (message, datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"), key))

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


def _row_to_lead(row: sqlite3.Row) -> Lead:
    return Lead(
        source=row["source"], source_id=row["source_id"], name=row["name"],
        category=row["category"], city=row["city"], address=row["address"],
        lat=row["lat"], lon=row["lon"], has_website=bool(row["has_website"]),
        website=row["website"], phone=row["phone"], email=row["email"],
        instagram=row["instagram"], vk=row["vk"], telegram=row["telegram"],
        score=row["score"], status=LeadStatus(row["status"]), message=row["message"],
        notes=row["notes"],
    )
