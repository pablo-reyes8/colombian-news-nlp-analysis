FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements/feature_extraction.txt /tmp/requirements.txt
COPY docker/requirements/common.txt /tmp/common.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN python -m spacy download es_core_news_sm

COPY . /app

CMD ["python", "scripts/pipeline/feature_extraction.py", "--help"]
