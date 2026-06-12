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
            llm_language_rule="""
എപ്പോഴും സാധാരണ സംസാരിക്കുന്ന ലളിതമായ മലയാളത്തിൽ മാത്രം മറുപടി നൽകുക.

പ്രധാന നിയമങ്ങൾ:
- കേരളത്തിലെ ഒരു പ്രൊഫഷണൽ ട്രാവൽ കൺസൾട്ടന്റിനെ പോലെ സംസാരിക്കുക.
- പുസ്തകങ്ങളിലെ ഭാഷയോ വിവർത്തനം ചെയ്തതുപോലുള്ള മലയാളമോ ഉപയോഗിക്കരുത്.
- ഒരിക്കലും ഈ വാക്കുകൾ ഉപയോഗിക്കരുത്: "നിങ്ങളുടെ പേര് നല്ലതാണ്", "നിങ്ങളുടെ ഇമെയിൽ നല്ലതാണ്", "സമൃദ്ധമായ സാംസ്കാരിക പൈതൃകം", "യാത്രാവിരലിന് തയ്യാറെടുക്കുന്ന".
- ഓരോ മറുപടിയും വളരെ ചെറുതായിരിക്കണം (പതിനഞ്ച് വാക്കുകളിൽ താഴെ).
- ഉപഭോക്താവിന്റെ വിവരങ്ങളെ വെറുതെ പുകഴ്ത്തരുത് (ഉദാഹരണത്തിന്: "നല്ല പേര്", "നല്ല ഇമെയിൽ" എന്ന് പറയരുത്).
- ലക്ഷ്യസ്ഥലത്തെക്കുറിച്ച് ഉപഭോക്താവ് ചോദിച്ചില്ലെങ്കിൽ നീണ്ട വിവരണം നൽകരുത്.
- ഒരു സമയം ഒരു ചോദ്യം മാത്രം ചോദിക്കുക.
- ഉപഭോക്താവിന്റെ മറുപടി ചെറുതായി അംഗീകരിച്ച് അടുത്ത ചോദ്യത്തിലേക്ക് പോവുക.

ഉദാഹരണങ്ങൾ:
ഉപഭോക്താവ്: എനിക്ക് തായ്‌ലൻഡിലേക്ക് പോകണം.
മറുപടി: ശരി. യാത്ര എത്ര ദിവസത്തേക്കാണ്? (പേരും ഇമെയിലും അറിയാമെങ്കിൽ നേരിട്ട് ദിവസം ചോദിക്കുക)

ഉപഭോക്താവ്: നാല് ദിവസം.
മറുപടി: ശരി. ഏത് തരത്തിലുള്ള താമസമാണ് താല്പര്യം? ബജറ്റ് ആണോ ലക്ഷ്വറി ആണോ?
""",
            uses_sarvam=True,
        )

    return VoiceLanguageConfig(
        code="en",
        label="English",
        pipecat_language=Language.EN_IN,
        greeting="Hi, I'm calling from Lifestyle Travels. Which destination are you planning to go to?",
        developer_hint="Ask one short question at a time and keep the conversation concise.",
        llm_language_rule="Always respond in English.",
        uses_sarvam=False,
    )


def build_system_instruction(
    lang: VoiceLanguageConfig,
    destination_catalog: str,
    greeting: str | None = None,
    customer_context: str | None = None,
    known_fields: dict | None = None,
    is_initial: bool = False,
) -> str:
    skip_hints: list[str] = []

    if known_fields:
        if known_fields.get("lead_name"):
            skip_hints.append(f"Name (value: '{known_fields['lead_name']}')")
        if known_fields.get("lead_email"):
            skip_hints.append(f"Email (value: '{known_fields['lead_email']}')")
        if known_fields.get("duration_days"):
            skip_hints.append(f"Duration (value: {known_fields['duration_days']} days)")
        if known_fields.get("accommodation"):
            skip_hints.append(f"Accommodation (value: '{known_fields['accommodation']}')")
        if known_fields.get("flight_needed") is not None:
            flight_label = "yes" if known_fields["flight_needed"] else "no"
            skip_hints.append(f"Flights (value: {flight_label})")

    skip_clause = ""
    if skip_hints:
        skip_clause = (
            " CRITICAL: The following fields are ALREADY KNOWN and MUST NOT be asked or confirmed again: "
            + ", ".join(skip_hints)
            + ". Skip them entirely. Do NOT say things like 'I have called you...'. Directly ask only for the remaining missing fields."
        )

    memory_clause = ""
    if customer_context:
        memory_clause = f"\n\nCustomer Memory:\n{customer_context}\n"

    malayalam_extra = ""
    if lang.code == "ml":
        malayalam_extra = (
            " MALAYALAM RULES: Speak like a real Kerala travel executive. "
            "Use simple spoken conversational Malayalam. "
            "Never use literary, textbook, poetic, or translated Malayalam. "
            "Do NOT praise or say things like 'നിങ്ങളുടെ പേര് നല്ലതാണ്', 'നിങ്ങളുടെ ഇമെയിൽ നല്ലതാണ്', 'സമൃദ്ധമായ സാംസ്കാരിക പൈതൃകം', 'യാത്രാവിരലിന് തയ്യാറെടുക്കുന്ന'. "
            "Keep replies under 10-15 words. Never repeat sentences. Ask only one question at a time. "
            "Understand that 'വേണ്ട' (veenda) or 'ആവശ്യമില്ല' (aavashyamilla) means 'No' (flight_needed=False). "
            "Understand that 'വേണം' (veenam) or 'അതെ' (athe) means 'Yes' (flight_needed=True). "
            "Understand that 'ബഡ്ജറ്റ്' (budget) means 'budget' accommodation tier, 'മിഡ് റേഞ്ച്' (mid-range) means 'mid-range', and 'ലക്ഷ്വറി' (luxury) means 'luxury'."
        )

    base = (
        "You are a professional travel consultant for Lifestyle Travels. "
        f"{lang.llm_language_rule} "
        f"{malayalam_extra} "
        "Collect these details one-by-one in strict order: (1) Destination, (2) Name, (3) Email, (4) Duration, (5) Accommodation class, (6) Flights. "
        "Ask ONLY ONE question at a time. "
        f"{skip_clause} "
        "Accept any destination. Catalog: " + destination_catalog + ". "
        "Call register_interest only after collecting all missing fields. "
        "Keep responses under 15 words. Use simple spoken language. "
        "Never repeat the user's name or details back. "
        "Do NOT repeat the initial greeting since the conversation is already underway."
    )

    return base + memory_clause