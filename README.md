# Travel Voice Agent (Pipecat)

This project runs a Pipecat-based realtime voice assistant using SmallWebRTC, Deepgram (STT/TTS), and Groq LLM.

Run locally with Docker Compose:

```bash
cp .env.example .env
# fill in DEEPGRAM_API_KEY and GROQ_API_KEY
docker compose up --build
```

Open: http://localhost:7860/client/
