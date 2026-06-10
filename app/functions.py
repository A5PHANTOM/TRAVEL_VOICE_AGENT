import os
import re
from typing import Any

from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from app.database import save_interest


def _offered_packages() -> set[str]:
    """Return a set of offered package names (casefolded) from env or defaults."""
    env = os.environ.get("OFFERED_PACKAGES", "Dubai,Goa,Thailand")
    items = [p.strip() for p in env.split(",") if p.strip()]
    return {p.casefold() for p in items}


def _normalize_text(value: str | None) -> str:
    return (value or "").strip()


def _is_placeholder_destination(value: str) -> bool:
    normalized = value.casefold().strip()
    normalized = re.sub(r"[^a-z0-9\s-]", "", normalized)
    if normalized in {
        "",
        "unknown",
        "not specified",
        "n/a",
        "na",
        "yes",
        "yeah",
        "yep",
        "yup",
        "sure",
        "ok",
        "okay",
        "hi",
        "hello",
        "hey",
        "lot",
        "then",
        "com",
        "code",
        "is",
    }:
        return True
    if "?" in value or len(value) > 50:
        return True
    if any(phrase in normalized for phrase in (
        "which location", "tell me", "can you", "what are", "hello",
        "planning for", "all the destination",
    )):
        return True
    return False


def _is_placeholder_name(value: str) -> bool:
    normalized = value.casefold().strip()
    normalized = re.sub(r"[^a-z\s-]", "", normalized)
    if normalized in {
        "",
        "unknown",
        "not specified",
        "n a",
        "na",
        "yes",
        "yeah",
        "yep",
        "hi",
        "hello",
        "hey",
        "lot",
        "com",
        "code",
        "is",
        "velda number",
        "number in velda",
    }:
        return True
    if "?" in value or len(value) > 50:
        return True
    return False


def _is_valid_email(value: str) -> bool:
    if not re.fullmatch(r"[\w\.-]+@[\w\.-]+\.\w+", value):
        return False
    local, _, domain = value.partition("@")
    if len(value) > 100 or len(local) > 64 or len(domain) > 64:
        return False
    # Reject common STT garbage (long runs of repeated characters or spaces).
    if re.search(r"(.)\1{4,}", local):
        return False
    if local.count(" ") > 1:
        return False
    return True


def _is_valid_accommodation(value: str) -> bool:
    return value.casefold() in {"budget", "mid-range", "mid range", "luxury"}


async def register_interest(
    params: FunctionCallParams,
    destination: str | None = None,
    package_type: str | None = None,
    duration_days: int | None = None,
    accommodation: str | None = None,
    flight_needed: bool | None = None,
    lead_name: str | None = None,
    lead_email: str | None = None,
    notes: str | None = None,
):
    """Save a travel lead from the conversation.

    Args:
        destination (str): The destination the user wants to travel to.
        package_type (str | None): Package theme, e.g. beach relaxation or adventure.
        duration_days (int | None): Number of travel days.
        accommodation (str | None): Accommodation tier, e.g. budget, luxury, or mid-range.
        flight_needed (bool | None): Whether flights should be included.
        lead_name (str | None): Name of the traveler.
        lead_email (str | None): Email for follow-up.
        notes (str | None): Any extra notes from the conversation.
    """
    destination_name = _normalize_text(destination)
    if _is_placeholder_destination(destination_name):
        await params.result_callback(
            {
                "status": "needs_destination",
                "message": "Please ask the client which destination they are planning to travel to.",
            }
        )
        return

    name = _normalize_text(lead_name)
    if _is_placeholder_name(name):
        await params.result_callback(
            {
                "status": "needs_name",
                "message": "Please ask the client for their name.",
            }
        )
        return

    email = _normalize_text(lead_email)
    if not _is_valid_email(email):
        await params.result_callback(
            {
                "status": "needs_email",
                "message": "Please ask the client for their email address.",
            }
        )
        return

    if duration_days is None or duration_days <= 0:
        await params.result_callback(
            {
                "status": "needs_duration",
                "message": "Please ask the client for their travel duration in days.",
            }
        )
        return

    accommodation_val = _normalize_text(accommodation)
    if not _is_valid_accommodation(accommodation_val):
        await params.result_callback(
            {
                "status": "needs_accommodation",
                "message": "Please ask the client for their accommodation class preference (budget, mid-range, or luxury).",
            }
        )
        return

    if flight_needed is None:
        await params.result_callback(
            {
                "status": "needs_flight",
                "message": "Please ask the client if they need flights included.",
            }
        )
        return

    record: dict[str, Any] = {
        "destination": destination_name,
        "package_type": package_type,
        "duration_days": duration_days,
        "accommodation": accommodation_val,
        "flight_needed": flight_needed,
        "lead_name": name,
        "lead_email": email,
        "notes": notes,
    }

    out_num = os.environ.get("outgoing_number") or os.environ.get("OUTGOING_NUMBER")
    if out_num:
        record["outgoing_number"] = out_num.strip()

    offered = _offered_packages()
    outside = destination_name.casefold() not in offered

    logger.info(f"Registering travel lead destination={destination_name} details={record}")

    try:
        # If there's an open partial lead (same name + package, no email), update it
        from app.database import find_open_lead, update_lead

        existing_id = await find_open_lead(destination_name, lead_name)
        if existing_id:
            row_id = await update_lead(existing_id, destination_name, record)
        else:
            row_id = await save_interest(destination_name, record)
        result = {
            "status": "ok",
            "id": row_id,
            "destination": destination_name,
            "outside_list": outside,
            "saved": record,
        }

        if outside:
            result["suggested_message"] = (
                f"Thanks — I’ve noted your interest in {destination_name}. Our executive will reach out to you shortly to help with options outside our standard packages."
            )

        await params.result_callback(result)
    except Exception as exc:
        logger.exception("Error saving interest")
        await params.result_callback({"status": "error", "error": str(exc)})
