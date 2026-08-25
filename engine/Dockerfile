# syntax=docker/dockerfile:1

# The CUDA runtime arrives through pip, not through the base image: torch
# resolves to a `+cuXXX` build on linux/amd64 and pulls the `nvidia-*` wheels
# with it, which is why they are deliberately absent from requirements.txt. Only
# the *driver* comes from the host, via nvidia-container-toolkit — so this stays
# a plain python image and the GPU is a compose concern.
#
# This is several GB. It is behind the `engine` compose profile for that reason:
# nothing but training and inference needs it.
FROM python:3.14-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS dev
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]

FROM base AS runtime
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
