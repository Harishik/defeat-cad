# DeFeat server image: the web app and API on port 8000, CPU only.
#   docker build -t defeat .
#   docker run -p 8000:8000 defeat
# For the assistant, run Ollama on the host (or another container) and point OLLAMA_HOST at it.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY defeat defeat
COPY models models
COPY samples samples
COPY web/dist web/dist
COPY launch.py .

RUN useradd --create-home defeat && mkdir -p /app/runs && chown defeat /app/runs
USER defeat
ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    DEFEAT_MAX_SESSIONS=24 \
    DEFEAT_MAX_UPLOAD_MB=200
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)"
CMD ["python", "-m", "uvicorn", "defeat.server:app", "--host", "0.0.0.0", "--port", "8000"]
