"""Extract and validate travel lead fields from conversation history."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from loguru import logger

from app.functions import (
    _is_placeholder_destination,
    _is_placeholder_name,
    _is_valid_accommodation,
    _is_valid_email,
)

EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")

DESTINATION_Q = re.compile(
    r"which destination are you planning|where are you planning to go|what destination",
    re.IGNORECASE,
)
NAME_Q = re.compile(
    r"what is your name|who am I speaking with|may I have your name|your name\?",
    re.IGNORECASE,
)
EMAIL_Q = re.compile(
    r"what is your email|email address|provide your email",
    re.IGNORECASE,
)
DURATION_Q = re.compile(
    r"how many days do you plan|how many days will you|how long do you plan|travel duration|days do you plan to travel|days will you be travel",
    re.IGNORECASE,
)
ACCOMMODATION_Q = re.compile(
    r"what type of accommodation|accommodation are you looking|class of accommodation|budget, mid-range, or luxury",
    re.IGNORECASE,
)
FLIGHT_Q = re.compile(
    r"do you need flights|flight requirements|flights included|need flights for your trip|flights for your trip",
    re.IGNORECASE,
)
DESTINATION_CONFIRM_Q = re.compile(
    r"^([A-Za-z][A-Za-z\s'-]{1,40}),\s*is that correct\??",
    re.IGNORECASE,
)
REGISTER_LEAK_RE = re.compile(r"register_interest\s*(\{.*\})", re.IGNORECASE | re.DOTALL)

AFFIRMATIVE = frozenset({
    "yes", "yeah", "yup", "correct", "that is correct", "thats correct",
    "yes that is correct", "yes correct", "yes please", "sure", "indeed",
    "that is right", "thats right", "right",
})
NEGATIVE = frozenset({
    "no", "nope", "not", "no thanks", "no thank you", "nay", "incorrect", "false",
    "i dont need", "i don't need", "dont need", "don't need",
})


def _known_destinations() -> list[str]:
    names: set[str] = set()
    env = os.environ.get("OFFERED_PACKAGES", "Dubai,Goa,Thailand")
    for item in env.split(","):
        item = item.strip()
        if item:
            names.add(item.casefold())
    knowledge_dir = os.path.join(os.getcwd(), "knowledge", "destinations")
    if os.path.isdir(knowledge_dir):
        for fn in os.listdir(knowledge_dir):
            if fn.endswith(".txt"):
                names.add(os.path.splitext(fn)[0].casefold())
    return sorted(names)


def _normalize_reply(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _is_affirmative(reply: str) -> bool:
    return _normalize_reply(reply) in AFFIRMATIVE


def _is_negative(reply: str) -> bool:
    normalized = _normalize_reply(reply)
    if normalized in NEGATIVE:
        return True
    return normalized.startswith("no ") or normalized.endswith(" dont need")


def _collect_user_replies(messages: list[dict[str, Any]], assistant_idx: int) -> list[str]:
    replies: list[str] = []
    for msg in messages[assistant_idx + 1 :]:
        if msg.get("role") == "assistant":
            break
        if msg.get("role") == "user":
            content = (msg.get("content") or "").strip()
            if content:
                replies.append(content)
    return replies


def _pick_destination_from_text(text: str) -> str | None:
    lower = text.casefold()
    for dest in _known_destinations():
        if re.search(rf"\b{re.escape(dest)}\b", lower):
            return dest.title() if dest.islower() else dest
    return None


def _pick_name(reply: str) -> str | None:
    if _is_affirmative(reply) or _is_negative(reply):
        return None
    name_match = re.search(
        r"(?:my name is|i am|i'm|this is|call me)\s+([A-Za-z][A-Za-z\s'-]{0,40})",
        reply,
        re.IGNORECASE,
    )
    candidate = name_match.group(1).strip() if name_match else reply.strip()
    candidate = re.sub(r"[.?!]+$", "", candidate).strip()
    if _is_placeholder_name(candidate) or len(candidate) > 50 or "?" in candidate:
        return None
    if not re.search(r"[A-Za-z]{2,}", candidate):
        return None
    return candidate


def _pick_email(replies: list[str]) -> str | None:
    for reply in replies:
        for email in EMAIL_RE.findall(reply):
            if _is_valid_email(email):
                return email
    return None


def _pick_duration(replies: list[str]) -> int | None:
    for reply in replies:
        match = re.search(r"\b(\d{1,3})\b", reply)
        if match:
            days = int(match.group(1))
            if 1 <= days <= 365:
                return days
        word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
        for word, num in word_map.items():
            if re.search(rf"\b{word}\b", reply, re.IGNORECASE):
                return num
    return None


def _pick_accommodation(replies: list[str]) -> str | None:
    for reply in replies:
        lower = reply.casefold()
        for option in ("luxury", "mid-range", "mid range", "budget"):
            if option in lower:
                return "mid-range" if "mid" in option else option
    return None


def _pick_flight_needed(replies: list[str]) -> bool | None:
    for reply in replies:
        if _is_affirmative(reply):
            return True
        if _is_negative(reply):
            return False
    return None


def _merge_tool_details(details: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("function", {}).get("name") != "register_interest":
                    continue
                try:
                    args = json.loads(tc["function"]["arguments"])
                except Exception:
                    continue
                for key in ("destination", "lead_name", "lead_email", "duration_days", "accommodation", "flight_needed"):
                    if args.get(key) is not None:
                        details[key] = args[key]

        content = msg.get("content")
        if msg.get("role") == "assistant" and isinstance(content, str):
            leak = REGISTER_LEAK_RE.search(content)
            if not leak:
                continue
            try:
                args = json.loads(leak.group(1))
            except Exception:
                continue
            for key in ("destination", "lead_name", "lead_email", "duration_days", "accommodation", "flight_needed"):
                if args.get(key) is not None:
                    details[key] = args[key]


def extract_details_from_history(messages: list[dict[str, Any]]) -> dict[str, Any]:
    details: dict[str, Any] = {
        "destination": None,
        "lead_name": None,
        "lead_email": None,
        "duration_days": None,
        "accommodation": None,
        "flight_needed": None,
    }

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        assistant_text = (msg.get("content") or "").strip()
        if not assistant_text:
            continue

        replies = _collect_user_replies(messages, i)
        if not replies:
            continue

        combined = " ".join(replies)

        confirm = DESTINATION_CONFIRM_Q.match(assistant_text)
        if confirm and _is_affirmative(replies[-1]):
            dest = confirm.group(1).strip()
            if not _is_placeholder_destination(dest):
                details["destination"] = dest

        if DESTINATION_Q.search(assistant_text):
            dest = _pick_destination_from_text(combined) or _pick_destination_from_text(replies[-1])
            if dest and not _is_placeholder_destination(dest):
                details["destination"] = dest
            elif replies[-1] and not _is_affirmative(replies[-1]) and not _is_negative(replies[-1]):
                candidate = replies[-1].strip().rstrip(".")
                if not _is_placeholder_destination(candidate) and len(candidate) <= 40 and "?" not in candidate:
                    details["destination"] = candidate

        if NAME_Q.search(assistant_text):
            for reply in replies:
                name = _pick_name(reply)
                if name:
                    details["lead_name"] = name
                    break

        if EMAIL_Q.search(assistant_text):
            email = _pick_email(replies)
            if email:
                details["lead_email"] = email

        if DURATION_Q.search(assistant_text):
            days = _pick_duration(replies)
            if days is not None:
                details["duration_days"] = days

        if ACCOMMODATION_Q.search(assistant_text):
            accommodation = _pick_accommodation(replies)
            if accommodation:
                details["accommodation"] = accommodation

        if FLIGHT_Q.search(assistant_text):
            flight = _pick_flight_needed(replies)
            if flight is not None:
                details["flight_needed"] = flight

    # Fallback: scan all user messages for a known destination.
    if not details["destination"]:
        for msg in messages:
            if msg.get("role") == "user" and msg.get("content"):
                dest = _pick_destination_from_text(msg["content"])
                if dest and not _is_placeholder_destination(dest):
                    details["destination"] = dest
                    break

    # Fallback: any valid email anywhere in the conversation.
    if not details["lead_email"]:
        for msg in messages:
            if msg.get("role") == "user" and msg.get("content"):
                for email in EMAIL_RE.findall(msg["content"]):
                    if _is_valid_email(email):
                        details["lead_email"] = email
                        break
            if details["lead_email"]:
                break

    _merge_tool_details(details, messages)
    return details


def should_save_partial_lead(details: dict[str, Any]) -> bool:
    dest = details.get("destination")
    name = details.get("lead_name")
    email = details.get("lead_email")

    has_dest = bool(dest) and not _is_placeholder_destination(str(dest))
    has_name = bool(name) and not _is_placeholder_name(str(name))
    has_email = bool(email) and _is_valid_email(str(email))

    if has_dest and (has_name or has_email):
        return True
    if has_name and has_email:
        return True
    return False


def build_partial_record(details: dict[str, Any], call_id: str | None) -> dict[str, Any] | None:
    if not should_save_partial_lead(details):
        return None

    destination = details.get("destination")
    if destination and _is_placeholder_destination(str(destination)):
        destination = None

    name = details.get("lead_name")
    if name and _is_placeholder_name(str(name)):
        name = None

    email = details.get("lead_email")
    if email and not _is_valid_email(str(email)):
        email = None

    duration = details.get("duration_days")
    if not isinstance(duration, int) or not (1 <= duration <= 365):
        duration = None

    accommodation = details.get("accommodation")
    if accommodation and not _is_valid_accommodation(str(accommodation)):
        accommodation = None

    flight = details.get("flight_needed")
    if not isinstance(flight, bool):
        flight = None

    if not destination and not (name and email):
        return None

    return {
        "destination": destination or "unknown",
        "package_type": None,
        "duration_days": duration,
        "accommodation": accommodation,
        "flight_needed": flight,
        "lead_name": name,
        "lead_email": email,
        "notes": f"Partially saved from disconnect/hangup/transfer. Call ID: {call_id or 'unknown'}",
    }
