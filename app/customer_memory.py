"""Customer memory: profiles, preferences, and interaction history."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from app.database import (
    backfill_customers_from_leads,
    get_customer_by_email,
    get_customer_by_phone,
    get_customer_with_interactions,
    save_interaction,
    upsert_customer,
)

BUDGET_FROM_ACCOMMODATION = {
    "budget": "Under ₹50,000",
    "mid-range": "₹50,000 – ₹1,50,000",
    "luxury": "Above ₹1,50,000",
}


def normalize_phone(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    digits = "".join(ch for ch in cleaned if ch.isdigit() or ch == "+")
    if digits.startswith("+"):
        return digits
    if digits:
        return f"+{digits}"
    return None


def accommodation_to_budget_range(accommodation: str | None) -> str | None:
    if not accommodation:
        return None
    key = accommodation.casefold().replace("mid range", "mid-range")
    return BUDGET_FROM_ACCOMMODATION.get(key)


def _format_destinations(destinations: list[str]) -> str:
    if not destinations:
        return "none recorded"
    return ", ".join(destinations)


def _interaction_summary(lead: dict[str, Any]) -> str:
    parts: list[str] = []
    dest = lead.get("destination")
    if dest:
        parts.append(str(dest))
    days = lead.get("duration_days")
    if days:
        parts.append(f"{days} days")
    accommodation = lead.get("accommodation")
    if accommodation:
        parts.append(str(accommodation))
    if lead.get("flight_needed") is True:
        parts.append("flights needed")
    elif lead.get("flight_needed") is False:
        parts.append("no flights")
    return "; ".join(parts) if parts else "travel inquiry"


async def lookup_customer_profile(
    phone: str | None = None,
    email: str | None = None,
) -> dict[str, Any] | None:
    """Find a customer by phone or email and return profile with interactions."""
    normalized_phone = normalize_phone(phone)
    normalized_email = (email or "").strip().lower() or None

    customer = None
    if normalized_phone:
        customer = await get_customer_by_phone(normalized_phone)
    if not customer and normalized_email:
        customer = await get_customer_by_email(normalized_email)
    if not customer:
        return None

    return await get_customer_with_interactions(customer["id"])


async def record_interaction_from_lead(
    lead_record: dict[str, Any],
    lead_id: int | None = None,
    call_id: str | None = None,
) -> int | None:
    """Upsert customer profile and append an interaction from a lead record."""
    phone = normalize_phone(lead_record.get("outgoing_number"))
    email = (lead_record.get("lead_email") or "").strip().lower() or None
    name = (lead_record.get("lead_name") or "").strip() or None

    if not phone and not email:
        logger.debug("Skipping customer memory: no phone or email on lead")
        return None

    destination = (lead_record.get("destination") or "").strip()
    accommodation = lead_record.get("accommodation")
    budget_range = accommodation_to_budget_range(accommodation) if accommodation else None

    destinations: list[str] = []
    if destination and destination.casefold() != "unknown":
        destinations.append(destination)

    preferences: dict[str, Any] = {}
    if lead_record.get("package_type"):
        preferences["package_type"] = lead_record["package_type"]
    if lead_record.get("notes"):
        preferences["last_notes"] = lead_record["notes"]

    customer_id = await upsert_customer(
        phone=phone,
        email=email,
        name=name,
        preferred_accommodation=accommodation,
        preferred_duration_days=lead_record.get("duration_days"),
        flight_preference=lead_record.get("flight_needed"),
        budget_range=budget_range,
        new_destinations=destinations,
        preferences_patch=preferences,
    )

    summary = _interaction_summary(lead_record)
    await save_interaction(
        customer_id=customer_id,
        lead_id=lead_id,
        call_id=call_id,
        summary=summary,
        destination=destination or None,
        duration_days=lead_record.get("duration_days"),
        accommodation=accommodation,
        flight_needed=lead_record.get("flight_needed"),
    )
    logger.info(f"Customer memory updated for customer_id={customer_id} summary={summary}")
    return customer_id


def known_fields_from_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    """Fields already on file that the agent may skip re-asking."""
    if not profile:
        return {}
    customer = profile.get("customer") or profile
    known: dict[str, Any] = {}
    if customer.get("name"):
        known["lead_name"] = customer["name"]
    if customer.get("email"):
        known["lead_email"] = customer["email"]
    if customer.get("preferred_duration_days"):
        known["duration_days"] = customer["preferred_duration_days"]
    if customer.get("preferred_accommodation"):
        known["accommodation"] = customer["preferred_accommodation"]
    if customer.get("flight_preference") is not None:
        known["flight_needed"] = customer["flight_preference"]
    destinations = customer.get("destinations") or []
    if destinations:
        known["previous_destinations"] = destinations
    return known


def build_personalization_context(profile: dict[str, Any] | None) -> str:
    """Build a system-prompt block for returning customers."""
    if not profile:
        return ""

    customer = profile.get("customer") or profile
    interactions = profile.get("interactions") or []

    lines = ["CUSTOMER MEMORY (returning client):"]
    if customer.get("name"):
        lines.append(f"- Name: {customer['name']}")
    if customer.get("phone"):
        lines.append(f"- Phone: {customer['phone']}")
    if customer.get("email"):
        lines.append(f"- Email: {customer['email']}")

    destinations = customer.get("destinations") or []
    if destinations:
        lines.append(f"- Previous destinations: {_format_destinations(destinations)}")

    if customer.get("preferred_duration_days"):
        lines.append(f"- Typical trip length: {customer['preferred_duration_days']} days")
    if customer.get("preferred_accommodation"):
        lines.append(f"- Accommodation preference: {customer['preferred_accommodation']}")
    if customer.get("budget_range"):
        lines.append(f"- Budget range: {customer['budget_range']}")
    if customer.get("flight_preference") is True:
        lines.append("- Usually needs flights included")
    elif customer.get("flight_preference") is False:
        lines.append("- Usually does not need flights")

    if interactions:
        recent = interactions[:3]
        summaries = [i.get("summary") or "inquiry" for i in recent]
        lines.append(f"- Recent interactions ({len(interactions)} total): {'; '.join(summaries)}")

    lines.append(
        "PERSONALIZATION RULES: Greet returning customers warmly by name if known. "
        "You may skip questions for name or email already on file unless the user wants to change them. "
        "Use their history to suggest similar destinations or ask if they want to revisit a previous destination. "
        "Still collect any missing required fields before calling register_interest."
    )
    return "\n".join(lines)


def build_returning_greeting(profile: dict[str, Any] | None, default_greeting: str) -> str:
    """Personalized opening greeting for a returning customer."""
    if not profile:
        return default_greeting

    customer = profile.get("customer") or profile
    name = customer.get("name")
    destinations = customer.get("destinations") or []

    if name and destinations:
        last_dest = destinations[-1]
        return (
            f"Hi {name}, welcome back to Lifestyle Travels! "
            f"I see you were interested in {last_dest} before. "
            f"Are you planning another trip to {last_dest}, or a different destination?"
        )
    if name:
        return (
            f"Hi {name}, welcome back to Lifestyle Travels! "
            "Which destination are you planning to go to this time?"
        )
    return default_greeting


async def ensure_customer_memory_seeded() -> int:
    """Backfill customer profiles from existing leads (idempotent)."""
    return await backfill_customers_from_leads()
