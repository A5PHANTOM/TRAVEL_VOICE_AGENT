FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential gcc ffmpeg git && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Install local pipecat and extras (webrtc, deepgram, groq)
RUN pip install --upgrade pip
RUN pip install -e ./pipecat[webrtc,deepgram,groq]

# Additional dependencies — faiss-cpu + scikit-learn replace chromadb entirely.
# No ONNX model downloads, no GPU packages, instant startup.
RUN pip install -r requirements.txt
RUN pip install aic-sdk

EXPOSE 7860

# Run the FastAPI app as a module so the `app` package is importable
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
