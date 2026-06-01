# Travel Voice Agent (Pipecat)

This project runs a Pipecat-based realtime voice assistant using SmallWebRTC, Deepgram (STT/TTS), and Groq LLM.

Run locally with Docker Compose:

```bash
cp .env.example .env
# fill in DEEPGRAM_API_KEY, GROQ_API_KEY, and TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN if using Twilio
docker compose up --build
```

Open: http://localhost:7860/client/

## Make an Outbound Call (Twilio)

To trigger an outbound call to a target phone number using Twilio:

1. Ensure `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM` (your Twilio phone number), and `PUBLIC_BASE_URL` (your public ngrok URL) are filled in your `.env` file.
2. Run the outbound call script:
   ```bash
   bash scripts/run_call.sh --to +1555XXXXXXX
   ```
