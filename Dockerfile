# KitPri v4 - Cooking Sound Detection (inference container)
#
# IMPORTANT: the INT8 model is quantized with the fbgemm (x86) engine.
# On Apple Silicon / ARM hosts, build & run with an explicit platform:
#   docker build --platform linux/amd64 -t kitpri:v4 .
#   docker run  --platform linux/amd64 --rm -v "$(pwd)/clips:/data" kitpri:v4 --audio /data/sample.wav
FROM --platform=linux/amd64 python:3.11-slim

# libsndfile is required by soundfile; ffmpeg allows non-wav input formats
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image small (no CUDA layers)
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY inference/ ./inference/

WORKDIR /app/inference

# Default: print help. Override with e.g.
#   docker run --rm -v $(pwd)/clips:/data kitpri:v4 --audio /data/sample.wav
ENTRYPOINT ["python", "predict.py"]
CMD ["--help"]
