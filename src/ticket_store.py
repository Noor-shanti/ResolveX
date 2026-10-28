"""Local SQLite persistence for requester-submitted support tickets."""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
from typing import Any
import uuid


TICKET_STATUSES = {"received", "in_review", "resolved", "escalated"}
STATUS_TRANSITIONS = {
    "received": {"received", "in_review", "escalated"},
    "in_review": {"in_review", "resolved", "escalated"},
    "resolved": {"resolved", "in_review"},
    "escalated": {"escalated", "in_review", "resolved"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class TicketStore:
    """A small local store; tracking credentials are stored only as hashes."""

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY,
                    tracking_token_hash TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    status_note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    analysis_json TEXT
                )
            """)
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_tickets_tracking_token_hash "
                "ON tickets (tracking_token_hash)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _public_fields(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "ticket_id": row["ticket_id"],
            "title": row["title"],
            "status": row["status"],
            "status_note": row["status_note"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _engineer_fields(row: sqlite3.Row) -> dict[str, Any]:
        item = TicketStore._public_fields(row)
        item["description"] = row["description"]
        item["analysis"] = json.loads(row["analysis_json"]) if row["analysis_json"] else None
        return item

    def create(self, title: str, description: str) -> tuple[dict[str, Any], str]:
        ticket_id = f"RX-{uuid.uuid4().hex[:10].upper()}"
        tracking_token = secrets.token_urlsafe(32)
        now = _now()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO tickets (ticket_id, tracking_token_hash, title, description, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ticket_id, _token_hash(tracking_token), title, description, "received", now, now),
            )
        return {
            "ticket_id": ticket_id,
            "title": title,
            "status": "received",
            "created_at": now,
            "updated_at": now,
        }, tracking_token

    def get_for_requester(self, tracking_token: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM tickets WHERE tracking_token_hash = ?",
                (_token_hash(tracking_token.strip()),),
            ).fetchone()
        return self._public_fields(row) if row else None

    def list_for_engineer(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM tickets ORDER BY CASE status WHEN 'received' THEN 0 WHEN 'in_review' THEN 1 ELSE 2 END, updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._engineer_fields(row) for row in rows]

    def get_for_engineer(self, ticket_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id.strip().upper(),)
            ).fetchone()
        return self._engineer_fields(row) if row else None

    def save_analysis(self, ticket_id: str, analysis: dict[str, Any]) -> dict[str, Any] | None:
        now = _now()
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE tickets SET analysis_json = ?, status = 'in_review', updated_at = ? WHERE ticket_id = ?",
                (json.dumps(analysis, ensure_ascii=False), now, ticket_id.strip().upper()),
            )
        return self.get_for_engineer(ticket_id) if cursor.rowcount else None

    def update_status(self, ticket_id: str, status: str, status_note: str = "") -> dict[str, Any] | None:
        if status not in TICKET_STATUSES:
            raise ValueError(f"Unsupported ticket status: {status}")
        current = self.get_for_engineer(ticket_id)
        if current is None:
            return None
        if status not in STATUS_TRANSITIONS[current["status"]]:
            raise ValueError(f"Cannot change ticket status from {current['status']} to {status}")
        now = _now()
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE tickets SET status = ?, status_note = ?, updated_at = ? WHERE ticket_id = ?",
                (status, status_note.strip(), now, ticket_id.strip().upper()),
            )
        return self.get_for_engineer(ticket_id) if cursor.rowcount else None
