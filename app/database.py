import json
import aiosqlite
from datetime import datetime, timezone
from typing import Any

DB_PATH = "/data/agents.db"


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


async def get_leads() -> list[dict]:
    db_path = DB_PATH
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT id, package, lead_json, timestamp FROM leads ORDER BY id DESC")
        rows = await cursor.fetchall()
    result = []
    for r in rows:
        result.append({"id": r[0], "package": r[1], "lead": json.loads(r[2] or "{}"), "timestamp": r[3]})
    return result
