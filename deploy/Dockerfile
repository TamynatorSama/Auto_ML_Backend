FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
COPY api/requirements.txt /tmp/api-requirements.txt
COPY worker/requirements.txt /tmp/worker-requirements.txt
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/requirements.txt -r /tmp/api-requirements.txt -r /tmp/worker-requirements.txt

COPY . /app
RUN useradd --uid 10001 --create-home automl \
    && mkdir -p /app/state \
    && chown automl:automl /app/state
USER automl
