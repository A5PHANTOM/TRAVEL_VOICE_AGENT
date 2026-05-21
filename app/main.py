import os
import asyncio
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from loguru import logger

from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
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
    logger.info("Starting Travel Voice Agent")

    api_key = _get_llm_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing LLM credentials. Set GROQ_API_KEY, OPENAI_API_KEY, or OPENAI_ADMIN_KEY."
        )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    stt = DeepgramSTTService(api_key=os.environ.get("DEEPGRAM_API_KEY", ""))
    tts = DeepgramTTSService(api_key=os.environ.get("DEEPGRAM_API_KEY", ""))

    llm = GroqLLMService(
        api_key=api_key,
        settings=GroqLLMService.Settings(
            system_instruction=(
                "You are a helpful travel assistant. Detect user intent and use function calls to "
                "register user interest in travel packages. Keep responses concise and natural for TTS. "
                "Do NOT display raw function-call syntax or slash-commands. Never show tool names or token-like text to the user. "
                "When you need to register interest, call the registered function directly and then speak a natural follow-up. "
                "Before ending a booking conversation, always collect the client's email address for follow-up. "
                "If the email is missing, ask for it in plain language and do not close the conversation until you have it. "
                "If you need to ask a clarifying question, do so in plain language (for example: 'I can register you for the Dubai luxury package — could you share your email address so I can send the details?')"
            )
        ),
    )

    # Register function that will persist the user's interest
    llm.register_direct_function(register_interest)

    # Build pipeline
    context = LLMContext(tools=ToolsSchema(standard_tools=[register_interest]))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off conversation
        context.add_message({"role": "developer", "content": "Hello! Ask me about our travel packages."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


@app.post("/api/offer")
@app.patch("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    return await _handle_webrtc_offer(request, background_tasks)


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

