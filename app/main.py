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

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import TTSSpeakFrame, LLMRunFrame, EndFrame, BotStoppedSpeakingFrame, Frame
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.services.llm_service import FunctionCallParams
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB on startup
    await init_db()
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
    await _run_agent(
        SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
        )
    )


async def _run_agent(transport: BaseTransport, caller_number: str | None = None, call_id: str | None = None):
    logger.info("Starting Travel Voice Agent")

    api_key = _get_llm_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing LLM credentials. Set GROQ_API_KEY, OPENAI_API_KEY, or OPENAI_ADMIN_KEY."
        )

    end_session_after_speaking = False

    class SessionEnder(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, BotStoppedSpeakingFrame) and end_session_after_speaking:
                logger.info("Ending session gracefully after transfer message spoken")
                await self.push_frame(EndFrame())

    session_ender = SessionEnder()

    async def transfer_to_human(params: FunctionCallParams):
        """Initiates a cold transfer to a human support agent when requested by the user."""
        nonlocal end_session_after_speaking
        logger.info("TRANSFER REQUESTED")
        logger.info(f"Caller: {caller_number}")

        if not caller_number:
            logger.error("No caller number available for transfer")
            await params.result_callback({
                "status": "error",
                "message": "No caller number available for transfer"
            })
            return

        account_sid = os.environ.get("EXOTEL_ACCOUNT_SID", "")
        exotel_api_key = os.environ.get("EXOTEL_API_KEY", "")
        exotel_api_token = os.environ.get("EXOTEL_API_TOKEN", "")
        support_number = os.environ.get("SUPPORT_NUMBER", "")
        caller_id = os.environ.get("CALLER_ID", "")

        if not all([account_sid, exotel_api_key, exotel_api_token, support_number, caller_id]):
            logger.error("Missing one or more Exotel transfer credentials/configuration environment variables")
            await params.result_callback({
                "status": "error",
                "message": "Transfer configuration error"
            })
            return

        url = f"https://api.exotel.com/v1/Accounts/{account_sid}/Calls/connect"
        auth = aiohttp.BasicAuth(exotel_api_key, exotel_api_token)
        payload = {
            "From": support_number,
            "To": caller_number,
            "CallerId": caller_id,
            "Record": "true"
        }

        try:
            async with session.post(url, auth=auth, data=payload) as response:
                response_text = await response.text()
                logger.info(f"Response: {response_text}")
                if response.status in (200, 201):
                    end_session_after_speaking = True
                    await params.result_callback({
                        "status": "success",
                        "message": "Transfer initiated successfully."
                    })
                else:
                    logger.error(f"Exotel transfer API failed with status {response.status}: {response_text}")
                    await params.result_callback({
                        "status": "failed",
                        "message": "Transfer API failed"
                    })
        except Exception as e:
            logger.exception(f"Error calling Exotel transfer API: {e}")
            await params.result_callback({
                "status": "error",
                "error": str(e)
            })

    async with aiohttp.ClientSession() as session:
        stt = DeepgramSTTService(api_key=os.environ.get("DEEPGRAM_API_KEY", ""))
        tts = DeepgramHttpTTSService(
            api_key=os.environ.get("DEEPGRAM_API_KEY", ""),
            settings=DeepgramHttpTTSService.Settings(
                voice="aura-2-andromeda-en",
            ),
            aiohttp_session=session,
        )

        llm = GroqLLMService(
            api_key=api_key,
            settings=GroqLLMService.Settings(
                system_instruction=(
                    "You are a helpful travel assistant for ABC Travels. Detect user intent and use function calls to "
                    "register user interest in travel packages. Keep responses concise, natural for TTS, and limited to one short sentence or one short question at a time. "
                    "Never stack multiple questions in a single reply. "
                    "Do NOT display raw function-call syntax or slash-commands. Never show tool names or token-like text to the user. "
                    "When you need to register interest, call the registered function directly and then speak a natural follow-up. "
                    "Before ending a booking conversation, always collect the client's email address for follow-up. "
                    "If the email is missing, ask for it in plain language and do not close the conversation until you have it. "
                    "If you need to ask a clarifying question, ask only one at a time in plain language. "
                    "If the user requests a human, customer support, representative, real person, or transfer, immediately call transfer_to_human(). Do not ask follow-up questions. After calling the function, say: 'Please hold while I connect you to a human agent.' "
                    "Open the conversation with: 'Hi ABC Travels. Which destination are you planning to go to?'"
                )
            ),
        )

        # Register functions
        llm.register_direct_function(register_interest)
        llm.register_direct_function(transfer_to_human)

        # Build pipeline
        context = LLMContext(tools=ToolsSchema(standard_tools=[register_interest, transfer_to_human]))
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer()
            ),
        )

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
            # Queue greeting frame directly to TTS service
            # This will synthesize and emit audio frames to the output transport
            logger.debug("Queueing greeting TTSSpeakFrame to TTS")
            await tts.queue_frame(TTSSpeakFrame("Hi, I'm calling from ABC Travels."))
            logger.debug("Greeting frame queued")
            context.add_message({"role": "developer", "content": "Ask one short question at a time and keep the conversation concise."})

            # After greeting, trigger LLM to generate opening question
            logger.debug("Queueing LLMRunFrame to trigger opening question")
            await llm.queue_frame(LLMRunFrame())
            logger.debug("LLMRunFrame queued")
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

        params = FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True)

        # _create_telephony_transport will set params.serializer appropriately
        transport = await _create_telephony_transport(websocket, params, transport_type, call_data)

        await _run_agent(transport)
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


@app.websocket("/exotel/media")
async def exotel_media(websocket: WebSocket):
    try:
        from pipecat.runner.utils import _create_telephony_transport, parse_telephony_websocket

        logger.info("Exotel voicebot websocket connected")

        transport_type, call_data = await parse_telephony_websocket(websocket)
        logger.info(f"Detected telephony provider: {transport_type}")

        if transport_type != "exotel":
            raise RuntimeError(f"Unexpected telephony provider for /exotel/media: {transport_type}")

        params = FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True)
        transport = await _create_telephony_transport(websocket, params, transport_type, call_data)

        caller_number = call_data.get("from")
        call_id = call_data.get("call_id")
        await _run_agent(transport, caller_number=caller_number, call_id=call_id)
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

