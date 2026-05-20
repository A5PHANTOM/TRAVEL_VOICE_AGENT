from pipecat.services.llm_service import FunctionCallParams
from loguru import logger
from app.database import save_interest
import os
from typing import Iterable


def _offered_packages() -> set[str]:
    """Return a set of offered package names (casefolded) from env or defaults."""
    env = os.environ.get("OFFERED_PACKAGES", "Dubai,Goa,Thailand")
    items = [p.strip() for p in env.split(",") if p.strip()]
    return {p.casefold() for p in items}


async def register_interest_handler(params: FunctionCallParams):
    """Handler for LLM function call 'register_interest'.

    Behavior:
    - Save the interest into the DB.
    - If the requested package is not in the offered list, mark the result with
      `outside_list: True` so the LLM can inform the user that an executive will reach out.

    Expects arguments like: {"package": "Dubai", "lead": {"name": "Alice", "email": "..."}}
    """
    package = params.arguments.get("package") or params.arguments.get("destination") or "unknown"
    lead = params.arguments.get("lead") or params.arguments

    logger.info(f"Registering interest for package={package} lead={lead}")

    offered = _offered_packages()
    outside = package.casefold() not in offered

    try:
        row_id = await save_interest(package, lead)
        result = {"status": "ok", "id": row_id, "package": package, "outside_list": outside}

        # Suggest a canned message to the LLM for fallback cases so the assistant
        # can tell the user an executive will reach out. The LLM may still produce
        # its own natural reply using this information.
        if outside:
            result["suggested_message"] = (
                f"Thanks — I've noted your interest in {package}. Our executive will reach out to you shortly to assist with options outside our standard packages."
            )

        await params.result_callback(result)
    except Exception as e:
        logger.exception("Error saving interest")
        await params.result_callback({"status": "error", "error": str(e)})
