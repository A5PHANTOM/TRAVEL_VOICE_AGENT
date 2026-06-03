import os
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
from pipecat.frames.frames import TTSSpeakFrame, LLMRunFrame, EndFrame, BotStoppedSpeakingFrame, Frame
from pipecat.services.llm_service import FunctionCallParams
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.serializers.twilio import TwilioFrameSerializer

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramHttpTTSService
from pipecat.services.groq.llm import GroqLLMService

from app.functions import register_interest
from app.database import get_leads, init_db
from app.api import router as api_router

load_dotenv(override=True)

try:
    from pipecat.audio.filters.aic_filter import AICFilter
    AIC_FILTER_AVAILABLE = True
except ImportError:
    AIC_FILTER_AVAILABLE = False

aic_license_key = os.environ.get("AIC_LICENSE_KEY", "")
aic_model_id = os.environ.get("AIC_MODEL_ID", "quail-vf-2.1-l-16khz")

import time
from pipecat.audio.vad.aic_vad import AICVADAnalyzer

class StartupProtectedAICVADAnalyzer(AICVADAnalyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = time.time()

    def voice_confidence(self, buffer: bytes) -> float:
        # Ignore VAD triggers (always return 0.0) during the first 1.5 seconds of the call
        if time.time() - self._start_time < 1.5:
            return 0.0
        return super().voice_confidence(buffer)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB on startup
    await init_db()

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

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request_data["sdp"],
            type=request_data["type"],
            restart_pc=request_data.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request_data["sdp"], type=request_data["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        background_tasks.add_task(run_bot, pipecat_connection)

    answer = pipecat_connection.get_answer()
    pcs_map[answer["pc_id"]] = pipecat_connection
    return answer


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    if not AIC_FILTER_AVAILABLE:
        raise RuntimeError("AICFilter library is not available.")
    if not aic_license_key:
        raise RuntimeError("AIC_LICENSE_KEY is not set.")

    try:
        aic_filter = AICFilter(
            license_key=aic_license_key,
            model_id=aic_model_id,
            enhancement_level=1.0,
        )
    except Exception as e:
        logger.error(f"Failed to initialize AICFilter: {e}")
        raise

    params = TransportParams(audio_in_enabled=True, audio_out_enabled=True)
    params.audio_in_filter = aic_filter

    await _run_agent(
        SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=params,
        ),
        aic_filter=aic_filter
    )


async def _run_agent(transport: BaseTransport, call_id: str | None = None, aic_filter: Any = None):
    logger.info("Starting Travel Voice Agent")

    api_key = _get_llm_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing LLM credentials. Set GROQ_API_KEY, OPENAI_API_KEY, or OPENAI_ADMIN_KEY."
        )

    async with aiohttp.ClientSession() as session:
        stt = DeepgramSTTService(api_key=os.environ.get("DEEPGRAM_API_KEY", ""))
        tts = DeepgramHttpTTSService(
            api_key=os.environ.get("DEEPGRAM_API_KEY", ""),
            settings=DeepgramHttpTTSService.Settings(
                voice="aura-2-andromeda-en",
            ),
            aiohttp_session=session,
        )

        end_session_after_speaking = False

        async def transfer_to_human(params: FunctionCallParams):
            """Connects the caller to a human agent/supervisor. Call this immediately when the user requests to speak to a representative, support, agent, or human."""
            nonlocal end_session_after_speaking
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

            # Extract conversation context summary from LLMContext
            destination = "Not specified"
            email = "Not specified"
            name = "Not specified"
            for msg in context.get_messages():
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        if tc.get("type") == "function" and tc.get("function", {}).get("name") == "register_interest":
                            try:
                                args = json.loads(tc["function"]["arguments"])
                                if args.get("destination"):
                                    destination = args.get("destination")
                                if args.get("lead_email"):
                                    email = args.get("lead_email")
                                if args.get("lead_name"):
                                    name = args.get("lead_name")
                            except Exception:
                                pass

            # Fallback to scanning messages for email if not set
            if email == "Not specified":
                import re
                email_regex = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
                for msg in context.get_messages():
                    if msg.get("role") == "user" and msg.get("content"):
                        matches = email_regex.findall(msg.get("content"))
                        if matches:
                            email = matches[0]
                            break

            summary = f"Destination: {destination}. Client Name: {name}. Email: {email}."
            logger.info(f"Generated warm transfer context: {summary}")

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
                f"<Number url=\"{whisper_url}\">{supervisor_number}</Number>"
                f"</Dial>"
                f"</Response>"
            )
            
            logger.info(f"Initiating redirect to {supervisor_number} on call {call_id} using URL {url} with TwiML screen URL {whisper_url}")
            
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

        llm = GroqLLMService(
            api_key=api_key,
            settings=GroqLLMService.Settings(
                model="llama-3.1-8b-instant",
                system_instruction=(
                    "You are a helpful travel assistant for Lifestyle Travels. "
                    "To gather client details, you must ask questions one by one in the following strict order: "
                    "(1) Destination, (2) Client Name, (3) Email Address, (4) Travel Duration in days, (5) Accommodation class (budget/mid-range/luxury), and (6) Flight requirements. "
                    "You must ask only one question at a time. Only ask the next question after the user has answered the previous one. "
                    "Do NOT ask for multiple details at once, and do NOT skip any steps in the order. "
                    "Do NOT call `register_interest` until you have gathered all six details. "
                    "Only after you have collected all six details, call `register_interest` to register the traveler's interest. "
                    "Keep responses concise, natural for text-to-speech, and limited to one short sentence or one short question at a time. "
                    "Never display raw function-call syntax or token-like text to the user. "
                    "If the user requests a human, customer support, representative, real person, or transfer at any point, immediately call transfer_to_human() and do not ask any more questions. After calling transfer_to_human(), say: 'Please hold while I connect you to a human agent.' "
                    "Open the conversation with: 'Hi, I'm calling from Lifestyle Travels. Which destination are you planning to go to?'"
                )
            ),
        )

        # Register functions
        llm.register_direct_function(register_interest)
        llm.register_direct_function(transfer_to_human)

        # Build pipeline
        context = LLMContext(tools=ToolsSchema(standard_tools=[register_interest, transfer_to_human]))
        
        # Setup VAD analyzer
        if not aic_filter:
            raise RuntimeError("AICFilter is required to initialize AICVADAnalyzer.")

        logger.info("Using StartupProtectedAICVADAnalyzer with AICFilter")
        vad_analyzer = StartupProtectedAICVADAnalyzer(
            vad_context_factory=lambda: aic_filter.get_vad_context(),
            speech_hold_duration=0.25,
            minimum_speech_duration=0.15,
            sensitivity=5.0,
        )

        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=vad_analyzer
            ),
        )

        class SessionEnder(FrameProcessor):
            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                await self.push_frame(frame, direction)
                if isinstance(frame, BotStoppedSpeakingFrame) and end_session_after_speaking:
                    logger.info("Gracefully ending AI session after hand-off message")
                    await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)

        session_ender = SessionEnder()

        pipeline = Pipeline([
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            session_ender,
            transport.output(),
            assistant_aggregator,
        ])

        task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True))

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            # Set initial context developer instructions and assistant greeting
            context.add_message({"role": "developer", "content": "Ask one short question at a time and keep the conversation concise."})
            context.add_message({"role": "assistant", "content": "Hi, I'm calling from Lifestyle Travels. Which destination are you planning to go to?"})

            # Wait briefly to ensure the pipeline task is fully ready and started
            await asyncio.sleep(0.2)

            # Queue greeting frame directly to TTS service to start conversation instantly without LLM lag
            logger.debug("Queueing greeting TTSSpeakFrame to TTS")
            await tts.queue_frame(TTSSpeakFrame("Hi, I'm calling from Lifestyle Travels. Which destination are you planning to go to?"))
            logger.debug("Greeting frame queued")
        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
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

        if not AIC_FILTER_AVAILABLE:
            raise RuntimeError("AICFilter library is not available.")
        if not aic_license_key:
            raise RuntimeError("AIC_LICENSE_KEY is not set.")

        try:
            aic_filter = AICFilter(
                license_key=aic_license_key,
                model_id=aic_model_id,
                enhancement_level=1.0,
            )
        except Exception as e:
            logger.error(f"Failed to initialize AICFilter: {e}")
            raise

        params = FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True)
        params.audio_in_filter = aic_filter

        # _create_telephony_transport will set params.serializer appropriately
        transport = await _create_telephony_transport(websocket, params, transport_type, call_data)

        call_id = call_data.get("call_id")
        await _run_agent(transport, call_id=call_id, aic_filter=aic_filter)
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
    from urllib.parse import urlparse

    parsed = urlparse(public)
    # Some deployments may provide a value without a netloc (e.g. missing scheme)
    # Fallback to parsed.path if netloc is empty. Strip whitespace to avoid accidental newlines.
    host = (parsed.netloc or parsed.path).strip().lstrip("/").rstrip("/")
    ws_scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
    websocket_url = f"{ws_scheme}://{host}/twilio/media"
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

        if not AIC_FILTER_AVAILABLE:
            raise RuntimeError("AICFilter library is not available.")
        if not aic_license_key:
            raise RuntimeError("AIC_LICENSE_KEY is not set.")

        try:
            aic_filter = AICFilter(
                license_key=aic_license_key,
                model_id=aic_model_id,
                enhancement_level=1.0,
            )
        except Exception as e:
            logger.error(f"Failed to initialize AICFilter: {e}")
            raise

        params = FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True)
        params.audio_in_filter = aic_filter

        transport = await _create_telephony_transport(websocket, params, transport_type, call_data)

        await _run_agent(transport, aic_filter=aic_filter)
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
    if "request_data" not in request_data and "requestData" not in request_data:
        request_data["request_data"] = active_sessions[session_id]

    return await _handle_webrtc_offer(request_data, background_tasks)


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
    active_sessions[session_id] = request_data.get("body", {})

    result: dict[str, Any] = {"sessionId": session_id}
    if request_data.get("enableDefaultIceServers"):
        result["iceConfig"] = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

    return result


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

