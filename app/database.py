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
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS transfers (
                call_id TEXT PRIMARY KEY,
                context TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT UNIQUE,
                email TEXT UNIQUE,
                name TEXT,
                preferred_accommodation TEXT,
                preferred_duration_days INTEGER,
                flight_preference INTEGER,
                budget_range TEXT,
                destinations_json TEXT,
                preferences_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                lead_id INTEGER,
                call_id TEXT,
                destination TEXT,
                duration_days INTEGER,
                accommodation TEXT,
                flight_needed INTEGER,
                summary TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
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


async def save_transfer_context(call_id: str, context: str) -> None:
    db_path = DB_PATH
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT OR REPLACE INTO transfers (call_id, context, timestamp) VALUES (?, ?, ?)",
            (call_id, context, now),
        )
        await db.commit()


async def get_transfer_context(call_id: str) -> str | None:
    db_path = DB_PATH
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT context FROM transfers WHERE call_id = ?",
            (call_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None


def _deserialize_customer_row(row: tuple) -> dict[str, Any]:
    destinations = json.loads(row[8] or "[]")
    preferences = json.loads(row[9] or "{}")
    return {
        "id": row[0],
        "phone": row[1],
        "email": row[2],
        "name": row[3],
        "preferred_accommodation": row[4],
        "preferred_duration_days": row[5],
        "flight_preference": None if row[6] is None else bool(row[6]),
        "budget_range": row[7],
        "destinations": destinations,
        "preferences": preferences,
        "created_at": row[10],
        "updated_at": row[11],
    }


async def get_customer_by_phone(phone: str) -> dict[str, Any] | None:
    db_path = DB_PATH
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            SELECT id, phone, email, name, preferred_accommodation, preferred_duration_days,
                   flight_preference, budget_range, destinations_json, preferences_json,
                   created_at, updated_at
            FROM customers WHERE phone = ?
            """,
            (phone,),
        )
        row = await cursor.fetchone()
    return _deserialize_customer_row(row) if row else None


async def get_customer_by_email(email: str) -> dict[str, Any] | None:
    db_path = DB_PATH
    normalized = email.strip().lower()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            SELECT id, phone, email, name, preferred_accommodation, preferred_duration_days,
                   flight_preference, budget_range, destinations_json, preferences_json,
                   created_at, updated_at
            FROM customers WHERE LOWER(email) = ?
            """,
            (normalized,),
        )
        row = await cursor.fetchone()
    return _deserialize_customer_row(row) if row else None


async def get_customer_by_id(customer_id: int) -> dict[str, Any] | None:
    db_path = DB_PATH
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            SELECT id, phone, email, name, preferred_accommodation, preferred_duration_days,
                   flight_preference, budget_range, destinations_json, preferences_json,
                   created_at, updated_at
            FROM customers WHERE id = ?
            """,
            (customer_id,),
        )
        row = await cursor.fetchone()
    return _deserialize_customer_row(row) if row else None


async def get_customer_interactions(customer_id: int, limit: int = 20) -> list[dict[str, Any]]:
    db_path = DB_PATH
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            SELECT id, customer_id, lead_id, call_id, destination, duration_days,
                   accommodation, flight_needed, summary, timestamp
            FROM interactions
            WHERE customer_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (customer_id, limit),
        )
        rows = await cursor.fetchall()

    return [
        {
            "id": r[0],
            "customer_id": r[1],
            "lead_id": r[2],
            "call_id": r[3],
            "destination": r[4],
            "duration_days": r[5],
            "accommodation": r[6],
            "flight_needed": None if r[7] is None else bool(r[7]),
            "summary": r[8],
            "timestamp": r[9],
        }
        for r in rows
    ]


async def get_customer_with_interactions(customer_id: int) -> dict[str, Any] | None:
    customer = await get_customer_by_id(customer_id)
    if not customer:
        return None
    interactions = await get_customer_interactions(customer_id)
    return {"customer": customer, "interactions": interactions}


async def list_customers(limit: int = 100) -> list[dict[str, Any]]:
    db_path = DB_PATH
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            SELECT id, phone, email, name, preferred_accommodation, preferred_duration_days,
                   flight_preference, budget_range, destinations_json, preferences_json,
                   created_at, updated_at
            FROM customers
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
    return [_deserialize_customer_row(r) for r in rows]


async def upsert_customer(
    *,
    phone: str | None = None,
    email: str | None = None,
    name: str | None = None,
    preferred_accommodation: str | None = None,
    preferred_duration_days: int | None = None,
    flight_preference: bool | None = None,
    budget_range: str | None = None,
    new_destinations: list[str] | None = None,
    preferences_patch: dict[str, Any] | None = None,
) -> int:
    db_path = DB_PATH
    now = datetime.now(timezone.utc).isoformat()
    normalized_email = email.strip().lower() if email else None

    existing = None
    if phone:
        existing = await get_customer_by_phone(phone)
    if not existing and normalized_email:
        existing = await get_customer_by_email(normalized_email)

    destinations = list(existing["destinations"]) if existing else []
    for dest in new_destinations or []:
        if dest and dest not in destinations:
            destinations.append(dest)

    preferences = dict(existing["preferences"]) if existing else {}
    if preferences_patch:
        preferences.update(preferences_patch)

    if existing:
        customer_id = existing["id"]
        merged_name = name or existing["name"]
        merged_phone = phone or existing["phone"]
        merged_email = normalized_email or existing["email"]
        merged_accommodation = preferred_accommodation or existing["preferred_accommodation"]
        merged_duration = (
            preferred_duration_days
            if preferred_duration_days is not None
            else existing["preferred_duration_days"]
        )
        merged_flight = (
            flight_preference
            if flight_preference is not None
            else existing["flight_preference"]
        )
        merged_budget = budget_range or existing["budget_range"]

        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                """
                UPDATE customers SET
                    phone = ?, email = ?, name = ?,
                    preferred_accommodation = ?, preferred_duration_days = ?,
                    flight_preference = ?, budget_range = ?,
                    destinations_json = ?, preferences_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    merged_phone,
                    merged_email,
                    merged_name,
                    merged_accommodation,
                    merged_duration,
                    None if merged_flight is None else int(merged_flight),
                    merged_budget,
                    json.dumps(destinations),
                    json.dumps(preferences),
                    now,
                    customer_id,
                ),
            )
            await db.commit()
        return customer_id

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            INSERT INTO customers (
                phone, email, name, preferred_accommodation, preferred_duration_days,
                flight_preference, budget_range, destinations_json, preferences_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                phone,
                normalized_email,
                name,
                preferred_accommodation,
                preferred_duration_days,
                None if flight_preference is None else int(flight_preference),
                budget_range,
                json.dumps(destinations),
                json.dumps(preferences or {}),
                now,
                now,
            ),
        )
        await db.commit()
        return cursor.lastrowid


async def save_interaction(
    *,
    customer_id: int,
    lead_id: int | None = None,
    call_id: str | None = None,
    summary: str,
    destination: str | None = None,
    duration_days: int | None = None,
    accommodation: str | None = None,
    flight_needed: bool | None = None,
) -> int:
    db_path = DB_PATH
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            INSERT INTO interactions (
                customer_id, lead_id, call_id, destination, duration_days,
                accommodation, flight_needed, summary, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                customer_id,
                lead_id,
                call_id,
                destination,
                duration_days,
                accommodation,
                None if flight_needed is None else int(flight_needed),
                summary,
                now,
            ),
        )
        await db.commit()
        return cursor.lastrowid


def _budget_from_accommodation(accommodation: str | None) -> str | None:
    if not accommodation:
        return None
    mapping = {
        "budget": "Under ₹50,000",
        "mid-range": "₹50,000 – ₹1,50,000",
        "luxury": "Above ₹1,50,000",
    }
    return mapping.get(accommodation.casefold().replace("mid range", "mid-range"))


async def backfill_customers_from_leads() -> int:
    """Build customer profiles from historical leads. Returns customers created/updated."""
    leads = await get_leads()
    count = 0
    for lead_row in leads:
        payload = lead_row.get("lead") or {}
        phone = payload.get("outgoing_number")
        email = payload.get("lead_email")
        if not phone and not email:
            continue

        if phone:
            phone = phone.strip()
            if not phone.startswith("+"):
                phone = f"+{''.join(ch for ch in phone if ch.isdigit())}"

        destination = payload.get("destination")
        destinations = []
        if destination and str(destination).casefold() != "unknown":
            destinations.append(destination)

        accommodation = payload.get("accommodation")
        await upsert_customer(
            phone=phone,
            email=email,
            name=payload.get("lead_name"),
            preferred_accommodation=accommodation,
            preferred_duration_days=payload.get("duration_days"),
            flight_preference=payload.get("flight_needed"),
            budget_range=_budget_from_accommodation(accommodation),
            new_destinations=destinations,
            preferences_patch={"package_type": payload.get("package_type")}
            if payload.get("package_type")
            else None,
        )

        customer = await get_customer_by_phone(phone) if phone else await get_customer_by_email(email)
        if not customer:
            continue

        lead_id = lead_row.get("id")
        summary_parts = []
        if destination:
            summary_parts.append(str(destination))
        if payload.get("duration_days"):
            summary_parts.append(f"{payload['duration_days']} days")
        if accommodation:
            summary_parts.append(str(accommodation))
        summary = "; ".join(summary_parts) or "travel inquiry"

        db_path = DB_PATH
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT id FROM interactions WHERE lead_id = ?",
                (lead_id,),
            )
            if await cursor.fetchone():
                continue

        await save_interaction(
            customer_id=customer["id"],
            lead_id=lead_id,
            call_id=None,
            summary=summary,
            destination=destination,
            duration_days=payload.get("duration_days"),
            accommodation=accommodation,
            flight_needed=payload.get("flight_needed"),
        )
        count += 1
    return count
