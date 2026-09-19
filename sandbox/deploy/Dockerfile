# The sandbox server's own image. Build context: sandbox/
FROM python:3.12-slim

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY automl_sandbox /app/automl_sandbox
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

EXPOSE 8765
# no access log: workers poll each exec twice a second
CMD ["uvicorn", "--factory", "automl_sandbox.app:from_config", "--host", "0.0.0.0", "--port", "8765", "--no-access-log"]
