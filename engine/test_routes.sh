#!/usr/bin/env bash
# Manual smoke tests for the engine's routes.
# Start the server first:  .venv/bin/uvicorn main:app --port 8001 --reload
set -euo pipefail

# 8000 is taken by an unrelated local Apache instance on this machine.
BASE_URL="${BASE_URL:-http://127.0.0.1:8001}"
MAX_BODY_CHARS="${MAX_BODY_CHARS:-800}"
MODEL_NAME="${MODEL_NAME:-test-model}"

hr() { printf -- '\n== %s ==\n' "$1"; }

# curl, printing the status code and a pretty-printed, truncated body.
req() {
    local method="$1" path="$2"
    shift 2
    local status body
    status="$(curl -sS -o /tmp/engine_body -w '%{http_code}' -X "$method" "$BASE_URL$path" "$@")"
    body="$(cat /tmp/engine_body)"

    if command -v jq >/dev/null 2>&1 && echo "$body" | jq -e . >/dev/null 2>&1; then
        body="$(echo "$body" | jq .)"
    fi
    if [ "${#body}" -gt "$MAX_BODY_CHARS" ]; then
        body="${body:0:$MAX_BODY_CHARS}... [truncated, ${#body} chars]"
    fi
    printf 'status: %s\n%s\n' "$status" "$body"
}

json() { req POST "$1" -H 'Content-Type: application/json' -d "$2"; }

hr "GET /"
req GET "/"

hr "POST /train (wait=true, blocks until done)"
json "/train?wait=true" "{
  \"model_name\": \"$MODEL_NAME\",
  \"dimensions\": 128,
  \"heads\": 4,
  \"layers\": 2,
  \"ctx_window\": 128,
  \"max_vocab\": 2048,
  \"epochs\": 1,
  \"batch_size\": 8,
  \"warmup_steps\": 20,
  \"rebuild_dataset\": true
}"

hr "POST /infer"
json /infer "{
  \"model_name\": \"$MODEL_NAME\",
  \"prompt\": \"The history of\",
  \"max_tokens\": 40
}"

hr "POST /infer/stream (SSE tokens)"
curl -sN -X POST "$BASE_URL/infer/stream" -H 'Content-Type: application/json' \
  -d "{\"model_name\": \"$MODEL_NAME\", \"prompt\": \"The history of\", \"max_tokens\": 12}" \
  | head -20

hr "GET /runs"
req GET "/runs"

# Progress streaming + cancellation, the two the job queue depends on:
#
#   RUN=$(curl -s -X POST "$BASE_URL/train" -H 'Content-Type: application/json' \
#         -d '{"model_name":"x","epochs":5}' | jq -r .run_id)
#   curl -sN "$BASE_URL/runs/$RUN/events"      # SSE progress frames
#   curl -s -X POST "$BASE_URL/runs/$RUN/cancel"
