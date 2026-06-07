# Travel Voice Agent (Pipecat)

This project runs a Pipecat-based realtime voice assistant using SmallWebRTC, Groq LLM, and multilingual STT/TTS:

- **English** — Deepgram
- **Hindi** — Sarvam AI
- **Malayalam** — Sarvam AI

Run locally with Docker Compose:

```bash
cp .env.example .env
# fill in DEEPGRAM_API_KEY (English), SARVAM_API_KEY (Hindi/Malayalam), GROQ_API_KEY
docker compose up --build
```

Set the voice language in `.env`:

```bash
VOICE_LANGUAGE=en   # English (default)
VOICE_LANGUAGE=hi   # Hindi (Sarvam)
VOICE_LANGUAGE=ml   # Malayalam (Sarvam)
```

Per-session override (WebRTC client): pass `"language": "hi"` in the `/start` request body.

Open: http://localhost:7860/client/

## Make an Outbound Call (Twilio)

To trigger an outbound call to a target phone number using Twilio:

1. Ensure `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM` (your Twilio phone number), and `PUBLIC_BASE_URL` (your public ngrok URL) are filled in your `.env` file.
2. Run the outbound call script:
   ```bash
   bash scripts/run_call.sh --to +1555XXXXXXX
   ```
