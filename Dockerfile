FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt


COPY . /app/

COPY chroma_db_med/ /data/chroma_db_med/

ENV DB_PATH=/data/chroma_db_med \
    INPUT_DIR=/data/input \
    MODEL_CACHE_DIR=/data/model_cache

RUN mkdir -p /data/input /data/model_cache

CMD ["tail", "-f", "/dev/null"]
