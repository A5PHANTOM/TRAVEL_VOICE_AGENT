"""Voice language configuration for English, Hindi, and Malayalam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pipecat.transcriptions.language import Language

VoiceLanguageCode = Literal["en", "hi", "ml"]

SUPPORTED_LANGUAGES: dict[VoiceLanguageCode, str] = {
    "en": "English",
    "hi": "Hindi",
    "ml": "Malayalam",
}


@dataclass(frozen=True)
class VoiceLanguageConfig:
    code: VoiceLanguageCode
    label: str
    pipecat_language: Language | None
    greeting: str
    developer_hint: str
    llm_language_rule: str
    uses_sarvam: bool


def normalize_language_code(value: str | None) -> VoiceLanguageCode:
    code = (value or "en").strip().lower().replace("_", "-")
    aliases = {
        "en": "en",
        "en-in": "en",
        "english": "en",
        "hi": "hi",
        "hi-in": "hi",
        "hindi": "hi",
        "ml": "ml",
        "ml-in": "ml",
        "malayalam": "ml",
    }
    resolved = aliases.get(code)
    if resolved in SUPPORTED_LANGUAGES:
        return resolved  # type: ignore[return-value]
    return "en"


def get_language_config(code: str | None) -> VoiceLanguageConfig:
    normalized = normalize_language_code(code)

    if normalized == "hi":
        return VoiceLanguageConfig(
            code="hi",
            label="Hindi",
            pipecat_language=Language.HI_IN,
            greeting=(
                "नमस्कार, मैं लाइफ़स्टाइल ट्रैवल्स से बोल रहा हूँ। "
                "आप किस गंतव्य की यात्रा की योजना बना रहे हैं?"
            ),
            developer_hint="एक समय में एक छोटा सवाल पूछें और बातचीत संक्षिप्त रखें।",
            llm_language_rule="हमेशा हिंदी में जवाब दें।",
            uses_sarvam=True,
        )

    if normalized == "ml":
        return VoiceLanguageConfig(
            code="ml",
            label="Malayalam",
            pipecat_language=Language.ML_IN,
            greeting=(
                "നമസ്കാരം, ലൈഫ്‌സ്റ്റൈൽ ട്രാവൽസിൽ നിന്നാണ് ഞാൻ വിളിക്കുന്നത്. "
                "നിങ്ങൾ ഏത് സ്ഥലത്തേക്കാണ് യാത്ര ചെയ്യാൻ ഉദ്ദേശിക്കുന്നത്?"
            ),
            developer_hint="ഒരു സമയം ഒരു ചെറിയ ചോദ്യം മാത്രം ചോദിക്കുക.",
            llm_language_rule="എപ്പോഴും മലയാളത്തിൽ മാത്രം മറുപടി നൽകുക.",
            uses_sarvam=True,
        )

    return VoiceLanguageConfig(
        code="en",
        label="English",
        pipecat_language=Language.EN_IN,
        greeting="Hi, I'm calling from Lifestyle Travels. Which destination are you planning to go to?",
        developer_hint="Ask one short question at a time and keep the conversation concise.",
        llm_language_rule="Always respond in English.",
        uses_sarvam=True,
    )


def build_system_instruction(lang: VoiceLanguageConfig, destination_catalog: str, greeting: str | None = None) -> str:
    base = (
        "You are a helpful travel assistant for Lifestyle Travels. "
        f"{lang.llm_language_rule} "
        "To gather client details, you must ask questions one by one in the following strict order: "
        "(1) Destination, (2) your name (ask 'What is your name?'), (3) Email Address, "
        "(4) Travel Duration in days, (5) Accommodation class (budget/mid-range/luxury), "
        "and (6) Flight requirements. "
        "You must ask only one question at a time. Only ask the next question after the user has answered the previous one. "
        "Do NOT ask for multiple details at once, and do NOT skip any steps in the order. "
        "Do NOT ask about booking packages, customizing trips, or other topics outside these six fields. "
        "Remember the destination the client already told you and do not switch to a different country unless they change it. "
        f"If asked what destinations are available, list: {destination_catalog}. "
        "Do NOT call `register_interest` until you have gathered all six details. "
        "The email must be a valid address with an @ symbol and domain (e.g. name@example.com). "
        "If the email sounds unclear, ask the user to repeat it. "
        "Only after you have collected all six details, call `register_interest` to register the traveler's interest. "
        "Once register_interest is successfully called, thank the client, inform them that an executive will reach out shortly, and conclude the conversation. "
        "Keep responses concise, natural for text-to-speech, and limited to one short sentence or one short question at a time. "
        "Never repeat the user's email, name, or other personal details back verbatim. Never repeat the same sentence twice. "
        "When the user asks about a destination, visa, package, or policy, use any 'Relevant knowledge' system messages "
        "to answer with one or two helpful facts about THEIR chosen destination, then ask the next unanswered required field. "
        "CRITICAL: Always invoke tools natively using the function-calling API. Never write function names, JSON arguments, "
        "or XML tags in your conversational text responses. "
        f"Open the conversation with: '{greeting if greeting is not None else lang.greeting}'. "
        "If the conversation is already underway, do NOT say the greeting again; instead, acknowledge the user's response in their language and ask the next question."
    )
    return base
