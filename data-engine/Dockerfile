# syntax=docker/dockerfile:1

# 3.14 matches the interpreter the checked-in venvs are built against;
# `pyproject.toml` allows anything from 3.13.
FROM python:3.14-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS dev
# Source arrives as a bind mount; the command lives in compose so the migration
# step is visible beside the service it belongs to.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]

FROM base AS runtime
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
