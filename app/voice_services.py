"""Factory for STT/TTS services based on voice language."""

from __future__ import annotations

import os
from typing import Any, Tuple

import aiohttp
from loguru import logger

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramHttpTTSService

from app.languages import VoiceLanguageConfig, get_language_config


def resolve_voice_language(language: str | None = None) -> VoiceLanguageConfig:
    return get_language_config(language or os.environ.get("VOICE_LANGUAGE", "en"))


def create_stt_tts(
    lang: VoiceLanguageConfig,
    session: aiohttp.ClientSession,
) -> Tuple[Any, Any]:
    """Return (stt, tts) for the given language configuration."""
    if lang.uses_sarvam:
        api_key = os.environ.get("SARVAM_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                f"SARVAM_API_KEY is required for {lang.label} (VOICE_LANGUAGE={lang.code}). "
                "Set it in your .env file."
            )

        from pipecat.services.sarvam.stt import SarvamSTTService
        from pipecat.services.sarvam.tts import SarvamHttpTTSService

        logger.info(f"Using Sarvam AI STT/TTS for {lang.label} ({lang.code})")
        stt = SarvamSTTService(
            api_key=api_key,
            settings=SarvamSTTService.Settings(
                model="saaras:v3",
                language=lang.pipecat_language,
                mode="transcribe",
            ),
        )
        tts = SarvamHttpTTSService(
            api_key=api_key,
            aiohttp_session=session,
            settings=SarvamHttpTTSService.Settings(
                model="bulbul:v3",
                language=lang.pipecat_language,
                speaker=_default_sarvam_speaker(lang.code),
            ),
        )
        return stt, tts

    deepgram_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not deepgram_key:
        raise RuntimeError("DEEPGRAM_API_KEY is required for English voice mode.")

    logger.info("Using Deepgram STT/TTS for English")
    stt = DeepgramSTTService(
        api_key=deepgram_key,
        settings=DeepgramSTTService.Settings(
            model="nova-2",
            smart_format=True,
            language="en",
        ),
    )
    tts = DeepgramHttpTTSService(
        api_key=deepgram_key,
        settings=DeepgramHttpTTSService.Settings(
            voice="aura-2-andromeda-en",
        ),
        aiohttp_session=session,
    )
    return stt, tts


def _default_sarvam_speaker(code: str) -> str:
    return {
        "hi": os.environ.get("SARVAM_HI_SPEAKER", "priya"),
        "ml": os.environ.get("SARVAM_ML_SPEAKER", "anushka"),
    }.get(code, "priya")
