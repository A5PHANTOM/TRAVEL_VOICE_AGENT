import os
import re
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import RedirectResponse, Response
from loguru import logger

from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import (
    TTSSpeakFrame,
    LLMRunFrame,
    LLMContextFrame,
    EndFrame,
    BotStoppedSpeakingFrame,
    Frame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSUpdateSettingsFrame,
    SystemFrame,
    ControlFrame,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.turns.user_start import VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.serializers.twilio import TwilioFrameSerializer

from pipecat.services.groq.llm import GroqLLMService

from app.functions import register_interest
from app.database import get_leads, init_db
from app.api import router as api_router
from app.rag import FaissRAG
from app.lead_extraction import (
    build_partial_record,
    extract_details_from_history,
    _known_destinations,
)
from app.languages import (
    SUPPORTED_LANGUAGES,
    VoiceLanguageConfig,
    build_system_instruction,
    get_language_config,
)
from app.customer_memory import (
    build_personalization_context,
    build_returning_greeting,
    ensure_customer_memory_seeded,
    known_fields_from_profile,
    lookup_customer_profile,
    normalize_phone,
    record_interaction_from_lead,
)
from app.voice_services import create_stt_tts, resolve_voice_language

load_dotenv(override=True)

try:
    from pipecat.audio.filters.aic_filter import AICFilter
    AIC_FILTER_AVAILABLE = True
except ImportError:
    AIC_FILTER_AVAILABLE = False

aic_license_key = "" # Force disabled to prevent model download overhead and echo/repetition loop bugs
aic_model_id = os.environ.get("AIC_MODEL_ID", "quail-vf-2.1-l-16khz")

import time
from pipecat.audio.vad.aic_vad import AICVADAnalyzer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

class StartupProtectedAICVADAnalyzer(AICVADAnalyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = None

    def voice_confidence(self, buffer: bytes) -> float:
        if self._start_time is None:
            self._start_time = time.time()
            logger.info("VAD audio processing started. Initializing 1.5s startup protection window.")

        # Ignore VAD triggers (always return 0.0) during the first 1.5 seconds of the call
        if time.time() - self._start_time < 1.5:
            return 0.0
        return super().voice_confidence(buffer)


class FallbackVADAnalyzer(VADAnalyzer):
    def __init__(self, primary_vad: VADAnalyzer, fallback_vad: VADAnalyzer):
        super().__init__(params=primary_vad.params)
        self.primary_vad = primary_vad
        self.fallback_vad = fallback_vad
        self.use_fallback = False

    def num_frames_required(self) -> int:
        if self.use_fallback:
            return self.fallback_vad.num_frames_required()
        return self.primary_vad.num_frames_required()

    def set_sample_rate(self, sample_rate: int):
        super().set_sample_rate(sample_rate)
        self.primary_vad.set_sample_rate(sample_rate)
        self.fallback_vad.set_sample_rate(sample_rate)

    def set_params(self, params: VADParams):
        super().set_params(params)
        self.primary_vad.set_params(params)
        self.fallback_vad.set_params(params)

    def voice_confidence(self, buffer: bytes) -> float:
        if not self.use_fallback:
            try:
                return self.primary_vad.voice_confidence(buffer)
            except Exception as e:
                logger.error(f"Primary VAD voice_confidence failed: {e}. Falling back to Silero VAD.")
                self.use_fallback = True
                self._vad_frames = self.fallback_vad.num_frames_required()
                self._vad_frames_num_bytes = self._vad_frames * self._num_channels * 2

        try:
            return self.fallback_vad.voice_confidence(buffer)
        except Exception as e:
            logger.error(f"Fallback VAD voice_confidence also failed: {e}")
            return 0.0

    async def cleanup(self):
        await self.primary_vad.cleanup()
        await self.fallback_vad.cleanup()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB on startup
    await init_db()
    try:
        seeded = await ensure_customer_memory_seeded()
        logger.info(f"Customer memory: backfilled {seeded} interaction(s) from existing leads")
    except Exception as e:
        logger.error(f"Customer memory backfill failed: {e}")

    # Pre-load/warm up the AIC model at startup to eliminate latency during calls
    if aic_license_key and AIC_FILTER_AVAILABLE:
        try:
            logger.info(f"Pre-loading and warming up AIC model: {aic_model_id}")
            from pipecat.audio.filters.aic_filter import AICModelManager
            from pathlib import Path
            model_download_dir = Path.home() / ".cache" / "pipecat" / "aic-models"
            # Acquire once to populate the singleton cache and keep reference count >= 1
            await AICModelManager.acquire(
                model_id=aic_model_id,
                model_download_dir=model_download_dir,
            )
            logger.info("AIC model pre-loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to pre-load AIC model: {e}")

    global _rag_store
    try:
        _rag_store = FaissRAG(
            knowledge_dir=os.path.join(os.getcwd(), "knowledge"),
            persist_dir=os.path.join(os.getcwd(), "faiss_db"),
            chunk_size=512,
            max_snippet_chars=250,
        )
        count = await asyncio.to_thread(_rag_store.ingest_knowledge)
        logger.info(f"RAG: ingested {count} knowledge chunks into FAISS at startup")
    except Exception as e:
        logger.error(f"Failed to initialize RAG index at startup: {e}")
        _rag_store = None

    # Pre-warm Silero VAD model at startup to eliminate per-call latency
    try:
        logger.info("Pre-warming Silero VAD model...")
        from pipecat.audio.vad.silero import SileroVADAnalyzer as _WarmVAD
        _warm_vad = _WarmVAD(params=VADParams())
        await _warm_vad.cleanup()
        del _warm_vad
        logger.info("Silero VAD model pre-warmed successfully.")
    except Exception as e:
        logger.warning(f"Failed to pre-warm Silero VAD model (will load on first call): {e}")

    yield
    # On shutdown disconnect any peer connections
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


app = FastAPI(lifespan=lifespan)
app.include_router(api_router, prefix="/api")
# Mount the prebuilt client UI
app.mount("/client", SmallWebRTCPrebuiltUI)

# Store active peer connections
pcs_map: dict[str, SmallWebRTCConnection] = {}
active_sessions: dict[str, dict[str, Any]] = {}
active_tasks: dict[str, asyncio.Task] = {}

# Shared RAG index — built once at app startup from knowledge/
_rag_store: FaissRAG | None = None

RAG_MARKER = "Relevant knowledge:"
RAG_INFO_KEYWORDS = (
    "tell me", "more about", "information", "what to see", "attractions",
    "places to", "visit", "visa", "package", "best time", "recommend", "suggest",
    "how much", "cost", "things to do", "overview", "describe", "details",
)
RAG_FILLER_UTTERANCES = frozenset({
    "hello", "hi", "hey", "huh", "yes", "no", "ok", "okay", "what", "pardon",
})
LEAKED_FUNCTION_TEXT_RE = re.compile(
    r"function\s*=\s*register_interest|"
    r"register_interest\s*[\{>]|"
    r"<\s*/?\s*function|"
    r'\{\s*"lead_name"\s*:',
    re.IGNORECASE,
)

ice_servers = [IceServer(urls=os.environ.get("STUN_SERVER", "stun:stun.l.google.com:19302"))]


def _get_llm_api_key() -> str:
    return (
        os.environ.get("GROQ_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("OPENAI_ADMIN_KEY", "")
    )


def _get_public_base_url(request: Request | None = None) -> str:
    public_base_url = (
        os.environ.get("PUBLIC_BASE_URL", "")
        or os.environ.get("TWILIO_PUBLIC_BASE_URL", "")
        or os.environ.get("BASE_URL", "")
    )
    if public_base_url:
        return public_base_url.rstrip("/")

    if request is None:
        raise RuntimeError("Missing PUBLIC_BASE_URL or TWILIO_PUBLIC_BASE_URL")

    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
    if not host:
        raise RuntimeError("Unable to determine public base URL for Twilio")
    return f"{scheme}://{host}".rstrip("/")


async def _handle_webrtc_offer(request_data: dict[str, Any], background_tasks: BackgroundTasks):
    pc_id = request_data.get("pc_id")
    sdp = request_data.get("sdp")
    offer_type = request_data.get("type")
    session_id = request_data.get("session_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]

        if not sdp:
            # PATCH without SDP = non-offer renegotiation signal (e.g. restart_pc
            # flag or track-status). Return the current answer so the client keeps
            # its connection alive without crashing.
            logger.debug(f"Renegotiation PATCH for {pc_id} has no sdp — returning current answer")
            answer = pipecat_connection.get_answer()
            return answer

        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=sdp,
            type=offer_type,
            restart_pc=request_data.get("restart_pc", False),
        )
    else:
        if not sdp:
            raise HTTPException(status_code=400, detail="Missing sdp in offer payload")

        # Clean up any existing connection for the same session_id before creating a new one
        if session_id:
            for old_pc_id, conn in list(pcs_map.items()):
                if getattr(conn, "_session_id", None) == session_id:
                    logger.info(f"Closing and removing old connection {old_pc_id} for session {session_id} on new connection request")
                    pcs_map.pop(old_pc_id, None)
                    try:
                        await conn.disconnect()
                    except Exception as e:
                        logger.error(f"Error disconnecting old connection: {e}")

                    # Cancel the associated bot task!
                    if old_pc_id in active_tasks:
                        logger.info(f"Cancelling bot task for old connection {old_pc_id}")
                        task = active_tasks.pop(old_pc_id)
                        task.cancel()

        pipecat_connection = SmallWebRTCConnection(ice_servers)
        if session_id:
            pipecat_connection._session_id = session_id
        
        # Save to pcs_map immediately so concurrent requests can find and terminate it
        pcs_map[pipecat_connection.pc_id] = pipecat_connection

        try:
            await pipecat_connection.initialize(sdp=sdp, type=offer_type)
        except Exception as e:
            pcs_map.pop(pipecat_connection.pc_id, None)
            raise

        # Check if this connection was discarded/replaced while we were initializing!
        if pipecat_connection.pc_id not in pcs_map:
            logger.warning(f"Connection {pipecat_connection.pc_id} was discarded during initialization. Not starting bot.")
            try:
                await pipecat_connection.disconnect()
            except Exception:
                pass
            raise HTTPException(status_code=409, detail="Connection was replaced during initialization")

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)
            # Cancel the associated bot task when the connection is closed
            if webrtc_connection.pc_id in active_tasks:
                logger.info(f"Cancelling bot task on connection close event for {webrtc_connection.pc_id}")
                task = active_tasks.pop(webrtc_connection.pc_id)
                task.cancel()

        voice_language = request_data.get("voice_language") or request_data.get("language")
        task = asyncio.create_task(run_bot(pipecat_connection, voice_language))
        active_tasks[pipecat_connection.pc_id] = task

    answer = pipecat_connection.get_answer()
    pcs_map[answer["pc_id"]] = pipecat_connection
    return answer


async def run_bot(webrtc_connection: SmallWebRTCConnection, voice_language: str | None = None):
    try:
        aic_filter = None
        if AIC_FILTER_AVAILABLE and aic_license_key:
            try:
                aic_filter = AICFilter(
                    license_key=aic_license_key,
                    model_id=aic_model_id,
                    enhancement_level=1.0,
                )
            except Exception as e:
                logger.error(f"Failed to initialize AICFilter: {e}")

        params = TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            audio_out_channels=1,
        )
        params.audio_in_filter = aic_filter

        await _run_agent(
            SmallWebRTCTransport(
                webrtc_connection=webrtc_connection,
                params=params,
            ),
            customer_phone=_resolve_customer_phone(),
            aic_filter=aic_filter,
            voice_language=voice_language,
        )
    finally:
        active_tasks.pop(webrtc_connection.pc_id, None)


async def save_partial_lead_from_history(messages: list[dict[str, Any]], call_id: str | None = None) -> None:
    # Check if register_interest was already successfully called in the history
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("content"):
            try:
                res = json.loads(msg["content"])
                if res.get("status") == "ok":
                    logger.info("Lead was already successfully registered. Skipping partial lead save.")
                    return
            except Exception:
                pass

    details = extract_details_from_history(messages)
    record = build_partial_record(details, call_id)
    if not record:
        logger.info("No valid conversational details gathered yet. Skipping partial lead save.")
        return

    from app.database import find_open_lead, update_lead, save_interest

    package_name = record["destination"]
    search_name = record["lead_name"]
    lead_id = None
    try:
        existing_id = await find_open_lead(package_name, search_name)
        if existing_id:
            logger.info(f"Updating existing partial lead {existing_id} with: {record}")
            lead_id = await update_lead(existing_id, package_name, record)
        else:
            logger.info(f"Saving new partial lead with: {record}")
            lead_id = await save_interest(package_name, record)

        try:
            await record_interaction_from_lead(record, lead_id=lead_id, call_id=call_id)
        except Exception as mem_exc:
            logger.warning(f"Customer memory update on partial lead failed: {mem_exc}")
    except Exception as e:
        logger.error(f"Failed to save partial lead: {e}")


class DynamicToolManager(FrameProcessor):
    def __init__(self, context: LLMContext, register_interest_tool: Any, transfer_to_human_tool: Any):
        super().__init__()
        self._context = context
        self._register_interest = register_interest_tool
        self._transfer_to_human = transfer_to_human_tool
        self._started = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._started = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, (SystemFrame, ControlFrame)):
            await self.push_frame(frame, direction)
        elif self._started:
            # Scan user messages for human transfer request keywords
            user_requested = False
            for msg in self._context.get_messages():
                if msg.get("role") == "user" and msg.get("content"):
                    content_lower = msg["content"].lower()
                    if any(kw in content_lower for kw in [
                        "human", "agent", "representative", "support", "supervisor",
                        "person", "operator", "transfer", "connect",
                        "speak", "talk", "someone", "somebody", "help desk",
                        "put me through", "customer care", "customer service"
                    ]):
                        user_requested = True
                        break

            # Dynamically set standard tools based on user request keywords presence
            from pipecat.adapters.schemas.tools_schema import ToolsSchema
            if user_requested:
                self._context.set_tools(ToolsSchema(standard_tools=[self._register_interest, self._transfer_to_human]))
            else:
                self._context.set_tools(ToolsSchema(standard_tools=[self._register_interest]))

            await self.push_frame(frame, direction)


class LanguageSwitcher(FrameProcessor):
    def __init__(
        self,
        context: LLMContext,
        stt: Any,
        tts: Any,
        session: aiohttp.ClientSession,
        destination_catalog: str,
        initial_lang: VoiceLanguageConfig,
        customer_context: str = "",
        known_fields: dict[str, Any] | None = None,
        llm: Any = None,
    ):
        super().__init__()
        self._context = context
        self._stt = stt
        self._tts = tts
        self._session = session
        self._destination_catalog = destination_catalog
        self._current_lang = initial_lang
        self._customer_context = customer_context
        self._known_fields = known_fields or {}
        self._started = False
        self._llm = llm

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._started = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, (SystemFrame, ControlFrame)):
            await self.push_frame(frame, direction)
        elif self._started:
            if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame):
                detected_lang_code = frame.language
                if detected_lang_code:
                    # Normalize language code to what our system expects (en, hi, ml)
                    from app.languages import normalize_language_code
                    resolved_code = normalize_language_code(detected_lang_code)

                    if resolved_code != self._current_lang.code:
                        logger.info(f"Language switch detected: {self._current_lang.code} -> {resolved_code}")
                        new_lang = get_language_config(resolved_code)
                        
                        # 1. Update LLM context with new system instructions
                        new_instructions = build_system_instruction(
                            new_lang,
                            self._destination_catalog,
                            customer_context=self._customer_context or None,
                            known_fields=self._known_fields or None,
                            is_initial=False,
                        )
                        
                        if self._llm:
                            self._llm._settings.system_instruction = (
                                new_instructions
                                + " IMPORTANT: Be natural and brief. Never repeat yourself or the same sentence twice in a row. If the user repeats themselves, acknowledge it naturally and move to the next question. Do NOT restart the greeting if the conversation is already underway."
                            )
                            logger.info("Successfully updated LLM service system instructions for language switch.")
                        
                        # Update system message in context by clearing any old switch system instructions
                        msgs = self._context.get_messages()
                        filtered_msgs = [
                            m for m in msgs
                            if not (
                                m.get("role") == "system"
                                and isinstance(m.get("content"), str)
                                and (
                                    m["content"].startswith("The user is now speaking")
                                    or "MALAYALAM RULES:" in m["content"]
                                    or "You are a professional travel consultant" in m["content"]
                                )
                            )
                        ]
                        self._context.set_messages(filtered_msgs)
                        self._context.add_message({
                            "role": "system",
                            "content": f"The user is now speaking {new_lang.label}. You MUST respond in {new_lang.label} only. Do NOT use any other language."
                        })

                        # 2. Update TTS service language if it's a MultilingualTTS
                        if hasattr(self._tts, "set_language"):
                            self._tts.set_language(resolved_code)

                        # 3. Update STT service language if it's a MultilingualSTTRouter
                        if hasattr(self._stt, "set_language"):
                            self._stt.set_language(resolved_code)
                        
                        self._current_lang = new_lang

            await self.push_frame(frame, direction)


def _resolve_customer_phone(call_data: dict[str, Any] | None = None) -> str | None:
    """Best-effort customer phone from telephony metadata or outbound env."""
    if call_data:
        for key in ("from_number", "from", "caller", "to_number", "to"):
            value = call_data.get(key)
            normalized = normalize_phone(value)
            if normalized:
                return normalized
    env_phone = os.environ.get("outgoing_number") or os.environ.get("OUTGOING_NUMBER")
    return normalize_phone(env_phone)


async def _run_agent(
    transport: BaseTransport,
    call_id: str | None = None,
    customer_phone: str | None = None,
    aic_filter: Any = None,
    voice_language: str | None = None,
):
    # Use multilingual mode by default if no specific language is requested, or if we want smart switching
    multilingual = True if voice_language is None else False
    lang = resolve_voice_language(voice_language)
    logger.info(f"Starting Travel Voice Agent (language={lang.code}, {lang.label}, multilingual={multilingual})")

    api_key = _get_llm_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing LLM credentials. Set GROQ_API_KEY, OPENAI_API_KEY, or OPENAI_ADMIN_KEY."
        )

    async with aiohttp.ClientSession() as session:
        stt, tts = create_stt_tts(lang, session, multilingual=multilingual)

        end_session_after_speaking = False
        lead_saved = False
        greeting_sent = False

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
            nonlocal end_session_after_speaking, lead_saved
            orig_callback = params.result_callback

            async def custom_callback(result: Any):
                nonlocal end_session_after_speaking, lead_saved
                if isinstance(result, dict) and result.get("status") == "ok":
                    end_session_after_speaking = True
                    lead_saved = True
                    logger.info("register_interest succeeded. Setting end_session_after_speaking = True to conclude call.")
                await orig_callback(result)

            params.result_callback = custom_callback
            from app.functions import register_interest as app_register_interest
            await app_register_interest(
                params=params,
                destination=destination,
                package_type=package_type,
                duration_days=duration_days,
                accommodation=accommodation,
                flight_needed=flight_needed,
                lead_name=lead_name,
                lead_email=lead_email,
                notes=notes,
            )

        async def transfer_to_human(params: FunctionCallParams):
            """Connects the caller to a human agent/supervisor. ONLY call this when the user explicitly and directly asks to speak to a human, representative, customer support, supervisor, or asks to transfer. Do NOT call this for regular conversation, questions, or greetings."""
            nonlocal end_session_after_speaking, lead_saved
            logger.info(f"TRANSFER REQUESTED for Call SID: {call_id}")
            if not call_id:
                logger.error("No active call_id available for transfer")
                await params.result_callback({"status": "failed", "message": "No active call ID found."})
                return

            account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
            auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
            supervisor_number = os.environ.get("SUPERVISOR_NUMBER")

            if not account_sid or not auth_token or not supervisor_number:
                logger.error("Missing Twilio credentials or supervisor number in environment")
                await params.result_callback({"status": "failed", "message": "Twilio configuration error."})
                return

            # Programmatic check: Ensure the user actually requested a transfer
            user_requested = False
            for msg in context.get_messages():
                if msg.get("role") == "user" and msg.get("content"):
                    content_lower = msg["content"].lower()
                    if any(kw in content_lower for kw in [
                        "human", "agent", "representative", "support", "supervisor", 
                        "real person", "operator", "transfer", "connect me", 
                        "speak to", "talk to", "someone", "somebody", "help desk", 
                        "put me through", "customer care"
                    ]):
                        user_requested = True
                        break
            
            if not user_requested:
                logger.warning("LLM attempted transfer_to_human, but no explicit user request was found in history. Rejecting tool call.")
                await params.result_callback({
                    "status": "ignored", 
                    "message": "Transfer not allowed. The user did not explicitly request a human agent, representative, or transfer. Do NOT transfer, and instead continue the conversation or say goodbye if the booking is complete."
                })
                return

            # Save partial lead details gathered so far
            if not lead_saved:
                try:
                    await save_partial_lead_from_history(context.get_messages(), call_id)
                    lead_saved = True
                except Exception as e:
                    logger.error(f"Failed to save partial lead during transfer: {e}")

            # Extract details for supervisor whisper
            details = extract_details_from_history(context.get_messages())
            destination = details.get("destination", "Not specified")
            name = details.get("lead_name", "Not specified")
            email = details.get("lead_email", "Not specified")

            summary = f"Destination: {destination}. Client Name: {name}. Email: {email}."
            logger.info(f"Generated warm transfer context: {summary}")

            # Smart routing: check for destination-specific supervisor number in environment
            target_supervisor = supervisor_number
            if destination and destination.lower() != "not specified":
                normalized_dest = destination.strip().lower()
                for env_key, env_val in os.environ.items():
                    clean_key = env_key.strip().lower()
                    if clean_key == f"{normalized_dest}_supervisor":
                        val = env_val.strip()
                        # Verify it's not a placeholder like +91xxxxxxxxx or +91xxxxxxxx
                        if val and not any(c in val.lower() for c in ["x", "placeholder"]):
                            target_supervisor = val
                            logger.info(f"Smart Routing: Found supervisor for destination '{destination}': {target_supervisor}")
                            break

            # Save the transfer context in the SQLite DB
            from app.database import save_transfer_context
            try:
                await save_transfer_context(call_id, summary)
            except Exception as db_err:
                logger.error(f"Failed to save transfer context to DB: {db_err}")

            # Twilio update call URL
            url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls/{call_id}.json"
            
            public_base_url = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
            whisper_url = f"{public_base_url}/twilio/whisper?call_id={call_id}"
            twiml_content = (
                f"<Response>"
                f"<Dial>"
                f"<Number url=\"{whisper_url}\">{target_supervisor}</Number>"
                f"</Dial>"
                f"</Response>"
            )
            
            logger.info(f"Initiating redirect to {target_supervisor} on call {call_id} using URL {url} with TwiML screen URL {whisper_url}")
            
            try:
                auth = aiohttp.BasicAuth(account_sid, auth_token)
                data = {
                    "Twiml": twiml_content
                }
                async with session.post(url, auth=auth, data=data) as resp:
                    resp_text = await resp.text()
                    logger.info(f"Twilio API Response Status: {resp.status}")
                    logger.info(f"Twilio API Response Body: {resp_text}")
                    if resp.status in (200, 201):
                        end_session_after_speaking = True
                        await params.result_callback({"status": "success", "message": "Transfer initiated successfully."})
                    else:
                        await params.result_callback({"status": "failed", "message": f"Twilio API failed with status {resp.status}."})
            except Exception as e:
                logger.exception(f"Error calling Twilio API: {e}")
                await params.result_callback({"status": "failed", "message": f"Exception during Twilio API call: {str(e)}"})

        customer_profile = await lookup_customer_profile(phone=customer_phone)
        if customer_profile:
            logger.info(
                f"Returning customer loaded: "
                f"{customer_profile.get('customer', {}).get('name') or 'unknown'} "
                f"({customer_profile.get('customer', {}).get('phone') or customer_phone})"
            )

        customer_context = build_personalization_context(customer_profile)
        known_fields = known_fields_from_profile(customer_profile)

        # Determine initial greeting — personalize for returning customers
        initial_greeting = build_returning_greeting(customer_profile, lang.greeting, lang.code)

        destination_catalog = ", ".join(d.title() for d in _known_destinations())
        default_model = "llama-3.3-70b-versatile"
        llm = GroqLLMService(
            api_key=api_key,
            settings=GroqLLMService.Settings(
                model=os.environ.get("GROQ_MODEL", default_model),
                temperature=0.65,
                extra={"frequency_penalty": 0.6, "presence_penalty": 0.6},
                system_instruction=build_system_instruction(
                    lang,
                    destination_catalog,
                    initial_greeting,
                    customer_context=customer_context or None,
                    known_fields=known_fields or None,
                    is_initial=True,
                )
                + " IMPORTANT: Be natural and brief. Never repeat yourself or the same sentence twice in a row. If the user repeats themselves, acknowledge it naturally and move to the next question. Do NOT restart the greeting if the conversation is already underway.",
            ),
        )

        # Register functions
        llm.register_direct_function(register_interest)
        llm.register_direct_function(transfer_to_human)

        rag = _rag_store

        # Build pipeline
        context = LLMContext(tools=ToolsSchema(standard_tools=[register_interest, transfer_to_human]))
        
        # Setup VAD analyzer
        primary_vad = None
        if aic_filter and aic_license_key:
            try:
                logger.info("Initializing primary VAD: StartupProtectedAICVADAnalyzer")
                primary_vad = StartupProtectedAICVADAnalyzer(
                    vad_context_factory=lambda: aic_filter.get_vad_context(),
                    speech_hold_duration=0.6,
                    minimum_speech_duration=0.15,
                    sensitivity=5.3,
                )
            except Exception as e:
                logger.error(f"Failed to initialize primary AIC VAD: {e}")

        logger.info("Initializing fallback VAD: SileroVADAnalyzer")
        fallback_vad = SileroVADAnalyzer(params=VADParams(min_volume=0.2, confidence=0.5))

        if primary_vad:
            logger.info("Using FallbackVADAnalyzer with StartupProtectedAICVADAnalyzer and Silero VAD.")
            vad_analyzer = FallbackVADAnalyzer(primary_vad, fallback_vad)
        else:
            logger.info("Running with Silero VAD directly.")
            vad_analyzer = fallback_vad

        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=vad_analyzer,
                user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
                user_turn_strategies=UserTurnStrategies(
                    start=[
                        VADUserTurnStartStrategy(enable_interruptions=True),
                        TranscriptionUserTurnStartStrategy(enable_interruptions=False),
                    ],
                    stop=[
                        SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.8),
                    ],
                ),
            ),
        )

        class SessionEnder(FrameProcessor):
            def __init__(self):
                super().__init__()
                self._started = False

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                if isinstance(frame, (SystemFrame, ControlFrame)):
                    await super().process_frame(frame, direction)

                if isinstance(frame, StartFrame):
                    self._started = True
                    await self.push_frame(frame, direction)
                    return

                if not self._started:
                    return

                if isinstance(frame, BotStoppedSpeakingFrame) and end_session_after_speaking:
                    logger.info("Gracefully ending AI session after hand-off message")
                    await self.push_frame(frame, direction)
                    await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
                else:
                    await self.push_frame(frame, direction)

        session_ender = SessionEnder()
        dynamic_tool_manager = DynamicToolManager(context, register_interest, transfer_to_human)
        language_switcher = LanguageSwitcher(
            context,
            stt,
            tts,
            session,
            destination_catalog,
            lang,
            customer_context=customer_context,
            known_fields=known_fields,
            llm=llm,
        )

        # Insert a small frame processor that injects relevant knowledge snippets
        # NOTE: FrameProcessor and FrameDirection are already imported at module level.
        # Do NOT re-import them here — a local import shadows the module-level name
        # and causes UnboundLocalError when SessionEnder (above) references it.

        class RAGAugmenter(FrameProcessor):
            def __init__(self, rag_client: FaissRAG, k: int = 3, min_score: float = 0.12):
                super().__init__()
                self.rag = rag_client
                self.k = k
                self.min_score = min_score
                self._started = False

            @staticmethod
            def _build_query(messages: list[dict]) -> tuple[str | None, str | None]:
                user_texts = [
                    m["content"].strip()
                    for m in messages
                    if m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip()
                ]
                if not user_texts:
                    return None, None
                last_user = user_texts[-1]
                lower = last_user.lower()
                # Only widen the query to recent history when the user is asking for info.
                if any(kw in lower for kw in RAG_INFO_KEYWORDS):
                    return last_user, " ".join(user_texts[-8:])
                return last_user, last_user

            @staticmethod
            def _should_inject(last_user: str, top_score: float, min_score: float) -> bool:
                normalized = re.sub(r"[^\w\s]", "", last_user.lower()).strip()
                if normalized in RAG_FILLER_UTTERANCES or len(normalized) < 4:
                    return False
                lower = last_user.lower()
                if not any(kw in lower for kw in RAG_INFO_KEYWORDS):
                    return False
                return top_score >= min_score

            @staticmethod
            def _scrub_leaked_assistant_text(context: LLMContext) -> None:
                msgs = context.get_messages()
                cleaned = [
                    m for m in msgs
                    if not (
                        m.get("role") == "assistant"
                        and isinstance(m.get("content"), str)
                        and LEAKED_FUNCTION_TEXT_RE.search(m["content"])
                    )
                ]
                if len(cleaned) != len(msgs):
                    context.set_messages(cleaned)
                    logger.warning("Removed leaked function-call text from assistant context")

            @staticmethod
            def _strip_old_injections(context: LLMContext) -> None:
                msgs = context.get_messages()
                filtered = [
                    m for m in msgs
                    if not (
                        m.get("role") == "system"
                        and isinstance(m.get("content"), str)
                        and m["content"].startswith(RAG_MARKER)
                    )
                ]
                if len(filtered) != len(msgs):
                    context.set_messages(filtered)

            async def process_frame(self, frame, direction: FrameDirection):
                if isinstance(frame, (SystemFrame, ControlFrame)):
                    await super().process_frame(frame, direction)

                if isinstance(frame, StartFrame):
                    self._started = True
                    await self.push_frame(frame, direction)
                    return

                if not self._started:
                    return

                # User aggregator emits LLMContextFrame downstream to trigger the LLM.
                if (
                    direction == FrameDirection.DOWNSTREAM
                    and isinstance(frame, LLMContextFrame)
                    and self.rag is not None
                ):
                    try:
                        context = frame.context
                        self._scrub_leaked_assistant_text(context)
                        last_user, query = self._build_query(context.get_messages())
                        if last_user and query:
                            snippets = await asyncio.to_thread(self.rag.get_relevant, query, self.k)
                            snippets = [s for s in snippets if s.get("score", 0) >= self.min_score]
                            if snippets and self._should_inject(last_user, snippets[0]["score"], self.min_score):
                                self._strip_old_injections(context)
                                builder = [RAG_MARKER]
                                for s in snippets:
                                    src = s.get("metadata", {}).get("source", "unknown")
                                    doc = s.get("document", "")
                                    builder.append(f"Source: {os.path.basename(src)}: {doc}")
                                context.add_message({"role": "system", "content": " \n ".join(builder)})
                                logger.info(
                                    f"RAG: injected {len(snippets)} snippet(s) "
                                    f"(top score {snippets[0]['score']:.2f}) for: {last_user[:80]!r}"
                                )
                    except Exception as e:
                        logger.warning(f"RAGAugmenter error: {e}")

                await self.push_frame(frame, direction)

        rag_augmenter = RAGAugmenter(rag) if rag is not None else None

        class AssistantTextSanitizer(FrameProcessor):
            """Clean LLM text that looks like a leaked tool call before it reaches TTS."""

            def __init__(self):
                super().__init__()
                self._started = False

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                if isinstance(frame, (SystemFrame, ControlFrame)):
                    await super().process_frame(frame, direction)

                if isinstance(frame, StartFrame):
                    self._started = True
                    await self.push_frame(frame, direction)
                    return

                if not self._started:
                    return

                if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TextFrame) and frame.text:
                    text = frame.text
                    # Remove XML-like function tags: <function=name>args</function>
                    text = re.sub(r"<[^>]*function[^>]*>.*?</[^>]*function[^>]*>", "", text, flags=re.IGNORECASE | re.DOTALL)
                    text = re.sub(r"<[^>]*function[^>]*>", "", text, flags=re.IGNORECASE)
                    text = re.sub(r"</[^>]*function[^>]*>", "", text, flags=re.IGNORECASE)
                    
                    # Remove raw JSON objects
                    text = re.sub(r"\{\s*\"[a-zA-Z_]+\"\s*:.*?\}", "", text, flags=re.DOTALL)
                    
                    # Remove lines containing function= or register_interest
                    lines = text.splitlines()
                    cleaned_lines = [line for line in lines if "function=" not in line.lower() and "register_interest" not in line.lower()]
                    text = "\n".join(cleaned_lines)
                    
                    frame.text = text

                await self.push_frame(frame, direction)

        assistant_text_sanitizer = AssistantTextSanitizer()

        pipeline_components = [
            transport.input(),
            stt,
            language_switcher,
            user_aggregator,
        ]

        if rag_augmenter:
            pipeline_components.append(rag_augmenter)

        pipeline_components += [
            dynamic_tool_manager,
            llm,
            assistant_text_sanitizer,
            tts,
            session_ender,
            transport.output(),
            assistant_aggregator,
        ]

        pipeline = Pipeline(pipeline_components)

        task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True))

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            # Set initial context developer instructions and assistant greeting
            if not greeting_sent:
                context.add_message({"role": "developer", "content": lang.developer_hint})
                context.add_message({"role": "assistant", "content": initial_greeting})

        @task.event_handler("on_pipeline_started")
        async def on_pipeline_started(task, frame):
            nonlocal greeting_sent
            if greeting_sent:
                logger.info("on_pipeline_started fired but greeting already sent")
                return

            logger.info("Pipeline started. Queueing greeting TTSSpeakFrame to TTS.")
            await tts.queue_frame(TTSSpeakFrame(initial_greeting))
            greeting_sent = True

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            nonlocal lead_saved
            logger.info(f"Client disconnected for call ID {call_id or 'unknown'}. Saving gathered details before cancelling.")
            if not lead_saved:
                try:
                    await save_partial_lead_from_history(context.get_messages(), call_id)
                    lead_saved = True
                except Exception as e:
                    logger.error(f"Error saving lead details on disconnect: {e}")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)


async def _run_twilio_bot(websocket: WebSocket):
    """Create a telephony transport by delegating parsing to pipecat's runner utils.

    Do NOT accept or consume websocket frames here; the FastAPIWebsocketTransport
    and serializers expect to handle the protocol-level messages themselves.
    """
    logger.info("Twilio media websocket connected")

    try:
        # Import runner utils which know how to parse telephony websockets and
        # construct a configured FastAPIWebsocketTransport with the appropriate
        # TwilioFrameSerializer (including stream/call ids).
        from pipecat.runner.utils import parse_telephony_websocket, _create_telephony_transport

        # Let the parser read the initial handshake messages and determine provider
        transport_type, call_data = await parse_telephony_websocket(websocket)

        aic_filter = None
        if AIC_FILTER_AVAILABLE and aic_license_key:
            try:
                aic_filter = AICFilter(
                    license_key=aic_license_key,
                    model_id=aic_model_id,
                    enhancement_level=1.0,
                )
            except Exception as e:
                logger.error(f"Failed to initialize AICFilter: {e}")

        params = FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True)
        params.audio_in_filter = aic_filter

        # _create_telephony_transport will set params.serializer appropriately
        transport = await _create_telephony_transport(websocket, params, transport_type, call_data)

        call_id = call_data.get("call_id")
        customer_phone = websocket.query_params.get("customer_phone")
        if not customer_phone:
            customer_phone = _resolve_customer_phone(call_data)
        else:
            customer_phone = normalize_phone(customer_phone)
        await _run_agent(
            transport,
            call_id=call_id,
            customer_phone=customer_phone,
            aic_filter=aic_filter,
            voice_language=os.environ.get("VOICE_LANGUAGE"),
        )
    except Exception as exc:
        logger.exception(f"Twilio agent error: {exc}")


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.post("/api/offer")
@app.patch("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    return await _handle_webrtc_offer(request, background_tasks)


@app.get("/twilio/voice")
@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    """Return TwiML that connects an incoming call to the Twilio media websocket."""
    # Twilio requires a WebSocket URL (ws:// or wss://) for Media Streams.
    public = _get_public_base_url(request)
    from urllib.parse import urlparse, quote

    parsed = urlparse(public)
    # Some deployments may provide a value without a netloc (e.g. missing scheme)
    # Fallback to parsed.path if netloc is empty. Strip whitespace to avoid accidental newlines.
    host = (parsed.netloc or parsed.path).strip().lstrip("/").rstrip("/")
    ws_scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"

    # Extract call parameters to identify customer phone number
    params = {}
    if request.method == "POST":
        try:
            form_data = await request.form()
            params = dict(form_data)
        except Exception:
            pass
    if not params:
        params = dict(request.query_params)

    from_num = params.get("From", "")
    to_num = params.get("To", "")
    direction = params.get("Direction", "")
    custom_to = params.get("to_number", "")
    if custom_to:
        to_num = custom_to

    twilio_from = os.environ.get("TWILIO_FROM") or os.environ.get("TWILIO_NUMBER") or ""
    
    # In outbound calls, customer is To; in inbound calls, customer is From.
    is_outbound = direction.startswith("outbound") or (
        twilio_from and normalize_phone(from_num) == normalize_phone(twilio_from)
    )
    customer_phone = to_num if is_outbound else from_num

    websocket_url = f"{ws_scheme}://{host}/twilio/media"
    if customer_phone:
        websocket_url += f"?customer_phone={quote(customer_phone)}"

    logger.debug(f"Twilio Stream URL: {websocket_url}")
    # Return TwiML that immediately instructs Twilio to open the Media Stream.
    twiml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        f"<Response>"
        f"<Connect><Stream url=\"{websocket_url}\" /></Connect>"
        f"</Response>"
    )
    return Response(content=twiml, media_type="application/xml")



@app.websocket("/twilio/media")
async def twilio_media(websocket: WebSocket):
    logger.info("Incoming Twilio websocket")
    await _run_twilio_bot(websocket)


@app.get("/twilio/whisper")
@app.post("/twilio/whisper")
async def twilio_whisper(request: Request):
    """Play a summary of the conversation to the supervisor before bridging the call."""
    call_id = request.query_params.get("call_id")
    logger.info(f"Twilio whisper requested for call_id: {call_id}")

    from app.database import get_transfer_context
    context_text = None
    if call_id:
        try:
            context_text = await get_transfer_context(call_id)
        except Exception as e:
            logger.error(f"Failed to load transfer context: {e}")

    if not context_text:
        context_text = "No conversation summary available."

    logger.info(f"Whispering context to supervisor: {context_text}")

    twiml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<Response>"
        f"<Say>Incoming warm transfer from Lifestyle Travels. Summary of conversation: {context_text}. Connecting you to the caller now.</Say>"
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/exotel/media")
async def exotel_media(websocket: WebSocket):
    try:
        from pipecat.runner.utils import _create_telephony_transport, parse_telephony_websocket

        logger.info("Exotel voicebot websocket connected")

        transport_type, call_data = await parse_telephony_websocket(websocket)
        logger.info(f"Detected telephony provider: {transport_type}")

        if transport_type != "exotel":
            raise RuntimeError(f"Unexpected telephony provider for /exotel/media: {transport_type}")

        aic_filter = None
        if AIC_FILTER_AVAILABLE and aic_license_key:
            try:
                aic_filter = AICFilter(
                    license_key=aic_license_key,
                    model_id=aic_model_id,
                    enhancement_level=1.0,
                )
            except Exception as e:
                logger.error(f"Failed to initialize AICFilter: {e}")

        params = FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True)
        params.audio_in_filter = aic_filter

        transport = await _create_telephony_transport(websocket, params, transport_type, call_data)

        customer_phone = _resolve_customer_phone(call_data)
        await _run_agent(
            transport,
            customer_phone=customer_phone,
            aic_filter=aic_filter,
        )
    except Exception as e:
        logger.exception(f"Exotel voicebot error: {e}")


@app.get("/twilio/media")
async def twilio_media_get(request: Request):
    """HTTP GET diagnostic for the Twilio media endpoint.

    Twilio or intermediaries may probe the WebSocket URL with a plain GET.
    Return a simple 200 response so these probes do not log 404s.
    The WebSocket upgrade will still be handled by the websocket route.
    """
    client_host = None
    try:
        client_host = request.client.host if request.client else None
    except Exception:
        client_host = None

    logger.info(f"HTTP GET to /twilio/media from {client_host or 'unknown'} - returning 200")
    return Response(
        content="This endpoint upgrades to WebSocket for Twilio Media Streams.",
        media_type="text/plain",
    )


@app.get("/exotel/media")
async def exotel_media_get(request: Request):
    """HTTP GET diagnostic for the Exotel media endpoint."""
    client_host = None
    try:
        client_host = request.client.host if request.client else None
    except Exception:
        client_host = None

    logger.info(f"HTTP GET to /exotel/media from {client_host or 'unknown'} - returning 200")
    return Response(
        content="This endpoint upgrades to WebSocket for Exotel voicebot media streams.",
        media_type="text/plain",
    )


@app.post("/sessions/{session_id}/api/offer")
@app.patch("/sessions/{session_id}/api/offer")
async def session_offer(session_id: str, request: Request, background_tasks: BackgroundTasks):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Invalid or not-yet-ready session_id")

    request_data = await request.json()
    logger.debug(f"session_offer [{request.method}] body keys: {list(request_data.keys())}")

    # RTVI wraps the actual WebRTC offer inside a `body` or `offer` key.
    # Unwrap it so _handle_webrtc_offer receives a dict with top-level sdp/type.
    offer_payload = (
        request_data.get("body")
        or request_data.get("offer")
        or request_data.get("requestData")
        or request_data.get("request_data")
    )
    if offer_payload and isinstance(offer_payload, dict) and "sdp" in offer_payload:
        # Merge pc_id from outer wrapper if present
        if "pc_id" in request_data and "pc_id" not in offer_payload:
            offer_payload["pc_id"] = request_data["pc_id"]
        request_data = offer_payload

    session_meta = active_sessions.get(session_id, {})
    existing_pc_id = session_meta.get("pc_id")

    if request.method == "POST":
        existing_pc_id = None
        session_meta.pop("pc_id", None)

    if existing_pc_id:
        logger.info(f"Mapping session {session_id} to existing connection {existing_pc_id}")
        request_data["pc_id"] = existing_pc_id

    request_data["session_id"] = session_id

    voice_language = session_meta.get("language") or request_data.get("language")
    if voice_language:
        request_data["voice_language"] = voice_language

    answer = await _handle_webrtc_offer(request_data, background_tasks)
    
    if answer and "pc_id" in answer and not existing_pc_id:
        session_meta["pc_id"] = answer["pc_id"]
        logger.info(f"Associated session {session_id} with connection {answer['pc_id']}")

    return answer


@app.post("/start")
async def start(request: Request):
    """Bootstrap endpoint for the prebuilt smallwebrtc client.

    Supports both runner-style bootstrap payloads and legacy SDP offers.
    """
    try:
        request_data = await request.json()
    except Exception:
        request_data = {}

    if "sdp" in request_data and "type" in request_data:
        return await _handle_webrtc_offer(request_data, BackgroundTasks())

    transport = request_data.get("transport", "webrtc")
    if transport != "webrtc":
        raise HTTPException(status_code=400, detail=f"Unsupported transport '{transport}'")

    session_id = request_data.get("sessionId") or request_data.get("session_id") or os.urandom(8).hex()
    session_body = dict(request_data.get("body", {}) or {})
    if request_data.get("language"):
        session_body["language"] = request_data["language"]
    active_sessions[session_id] = session_body

    result: dict[str, Any] = {"sessionId": session_id}
    if request_data.get("enableDefaultIceServers"):
        result["iceConfig"] = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    return result


@app.get("/api/languages")
async def api_languages():
    """Supported voice languages and active default."""
    default = get_language_config(os.environ.get("VOICE_LANGUAGE"))
    return {
        "default": default.code,
        "languages": [
            {"code": code, "label": label, "provider": "sarvam" if code != "en" else "deepgram"}
            for code, label in SUPPORTED_LANGUAGES.items()
        ],
    }


@app.get("/report")
async def report():
    """Return all saved travel lead reports."""
    return {"reports": await get_leads()}


@app.get("/api/reports")
async def api_reports():
    """API alias for the saved reports list."""
    return {"reports": await get_leads()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
