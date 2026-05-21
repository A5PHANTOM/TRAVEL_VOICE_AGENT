import os
from typing import Any

from loguru import logger

from pipecat.services.llm_service import FunctionCallParams

from app.database import save_interest


def _offered_packages() -> set[str]:
    """Return a set of offered package names (casefolded) from env or defaults."""
    env = os.environ.get("OFFERED_PACKAGES", "Dubai,Goa,Thailand")
    items = [p.strip() for p in env.split(",") if p.strip()]
    return {p.casefold() for p in items}


async def register_interest(
    params: FunctionCallParams,
    destination: str,
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
    destination_name = destination.strip() or "unknown"
    email = (lead_email or "").strip()
    if not email:
        await params.result_callback(
            {
                "status": "needs_email",
                "destination": destination_name,
                "message": "Please ask the client for their email address before ending the conversation or saving the booking.",
            }
        )
        return

    record: dict[str, Any] = {
        "destination": destination_name,
        "package_type": package_type,
        "duration_days": duration_days,
        "accommodation": accommodation,
        "flight_needed": flight_needed,
        "lead_name": lead_name,
        "lead_email": email,
        "notes": notes,
    }

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
