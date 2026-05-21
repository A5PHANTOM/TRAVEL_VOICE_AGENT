import json
import os
import aiosqlite
from datetime import datetime, timezone
from typing import Any

DB_PATH = os.environ.get("DB_PATH", "/data/agents.db")


async def init_db(path: str | None = None) -> None:
    db_path = path or DB_PATH
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                package TEXT NOT NULL,
                lead_json TEXT,
                timestamp TEXT NOT NULL
            )
            """
        )
        await db.commit()


async def save_interest(package: str, lead: dict[str, Any] | None = None) -> int:
    db_path = DB_PATH
    now = datetime.now(timezone.utc).isoformat()
    lead_json = json.dumps(lead or {})
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "INSERT INTO leads (package, lead_json, timestamp) VALUES (?, ?, ?)",
            (package, lead_json, now),
        )
        await db.commit()
        return cursor.lastrowid


async def find_open_lead(package: str, lead_name: str | None) -> int | None:
    """Find an existing lead for the same package with no email recorded.

    Returns the `id` of a matching lead or None if not found.
    This does a small recent scan and inspects the JSON payload to avoid
    inserting repeated partial records when the assistant retries.
    """
    if not lead_name:
        return None

    db_path = DB_PATH
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT id, lead_json FROM leads WHERE package = ? ORDER BY id DESC LIMIT 50",
            (package,)
        )
        rows = await cursor.fetchall()

    for r in rows:
        try:
            payload = json.loads(r[1] or "{}")
        except Exception:
            continue
        name = payload.get("lead_name")
        email = payload.get("lead_email")
        if name and name == lead_name and not email:
            return int(r[0])
    return None


async def update_lead(lead_id: int, package: str, lead: dict[str, Any] | None = None) -> int:
    """Update an existing lead row's package, JSON payload and timestamp."""
    db_path = DB_PATH
    now = datetime.now(timezone.utc).isoformat()
    lead_json = json.dumps(lead or {})
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE leads SET package = ?, lead_json = ?, timestamp = ? WHERE id = ?",
            (package, lead_json, now, lead_id),
        )
        await db.commit()
    return lead_id


async def get_leads() -> list[dict]:
    db_path = DB_PATH
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT id, package, lead_json, timestamp FROM leads ORDER BY id DESC")
        rows = await cursor.fetchall()
    result = []
    for r in rows:
        result.append({"id": r[0], "package": r[1], "lead": json.loads(r[2] or "{}"), "timestamp": r[3]})
    return result
