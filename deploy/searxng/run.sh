#!/usr/bin/env bash
# The same container docker-compose.yml describes, for a machine without the
# compose plugin (it is a separate package on most distributions).
#
#   ./run.sh          start it, generating a secret if none is set
#   ./run.sh stop     stop and remove it
#   ./run.sh logs     follow the log
#
# Prepend `sudo` if your user is not in the `docker` group. Being in that group
# is root-equivalent, so choosing between the two is a decision for whoever owns
# the machine rather than something this script should make.

set -euo pipefail

NAME=slate-searxng
IMAGE=docker.io/searxng/searxng:latest
# 8888 because this project already holds 8080 (the backend), 8001 (the engine)
# and 3000 (the UI) — see the port map in the root README.
PORT="${SEARXNG_PORT:-8888}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-start}" in
stop)
    docker rm -f "$NAME"
    exit 0
    ;;
logs)
    docker logs -f "$NAME"
    exit 0
    ;;
start) ;;
*)
    echo "usage: $0 [start|stop|logs]" >&2
    exit 2
    ;;
esac

# A fresh one each run unless the caller pinned it. Nothing here is reachable
# from outside the host, so a rotating secret costs nothing.
SEARXNG_SECRET="${SEARXNG_SECRET:-$(openssl rand -hex 32)}"

mkdir -p "$HERE/config" "$HERE/data"

docker rm -f "$NAME" >/dev/null 2>&1 || true

# Checked here because Docker reports it as a wall of endpoint-programming text
# that never names the port's actual occupant — and on this project the likely
# occupant is one of our own services.
if command -v ss >/dev/null && ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
    echo "Port $PORT is already in use. Set SEARXNG_PORT to something else:" >&2
    echo "  SEARXNG_PORT=8889 $0" >&2
    exit 1
fi

# Flags mirror docker-compose.yml; keep the two in step. Loopback-only because
# an instance with its limiter off will be scraped by anyone who can reach it.
docker run -d \
    --name "$NAME" \
    --restart unless-stopped \
    -p "127.0.0.1:$PORT:8080" \
    -v "$HERE/config:/etc/searxng:rw" \
    -v "$HERE/data:/var/cache/searxng:rw" \
    -e "SEARXNG_BASE_URL=http://localhost:$PORT/" \
    -e "SEARXNG_SECRET=$SEARXNG_SECRET" \
    --cap-drop ALL \
    --cap-add CHOWN --cap-add SETGID --cap-add SETUID \
    --log-opt max-size=1m --log-opt max-file=1 \
    "$IMAGE"

echo "Waiting for $NAME to answer…"
for _ in $(seq 30); do
    if curl -fsS "http://localhost:$PORT/search?q=test&format=json" >/dev/null 2>&1; then
        echo "ready — set SEARCH_URL=http://localhost:$PORT in backend/.env"
        exit 0
    fi
    sleep 1
done

# The failure worth naming: the instance is up but refusing the API, which is
# `json` missing from `search.formats` or the limiter being on. Both live in
# config/settings.yml and both look identical from here — a 403.
echo "Started, but the JSON API did not answer in 30s. Check:" >&2
echo "  docker logs $NAME" >&2
echo "  curl -i 'http://localhost:$PORT/search?q=test&format=json'" >&2
echo "A 403 means config/settings.yml did not take effect." >&2
exit 1
