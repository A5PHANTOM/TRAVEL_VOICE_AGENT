# Travel Voice Agent (Pipecat)

This project runs a Pipecat-based realtime voice assistant using SmallWebRTC, Deepgram (STT/TTS), and Groq LLM.

Run locally with Docker Compose:

```bash
cp .env.example .env
# fill in DEEPGRAM_API_KEY, GROQ_API_KEY, and TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN if using Twilio
docker compose up --build
```

Open: http://localhost:7860/client/
