#!/usr/bin/env bash
# The sandboxed shell behind the backend's `bash` tool.
#
#   ./run.sh          install if needed, then start it in the background
#   ./run.sh stop     stop it
#   ./run.sh logs     follow the log
#   ./run.sh fg       run in the foreground (Ctrl-C to stop)
#
# No container, unlike deploy/searxng: just-bash is the sandbox. It simulates
# bash over a virtual filesystem in-process, so there is nothing here that a
# container would be isolating — `rm -rf /` empties a Map. What it does need is
# Node ≥ 20.18.1.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 8899 because this project already holds 8080 (backend), 8001 (engine), 8002
# (data engine), 8888 (SearXNG) and 3000 (UI) — see the port map in the root
# README.
PORT="${BASH_PORT:-8899}"
PIDFILE="$HERE/.pid"
LOGFILE="$HERE/server.log"

stop_it() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        kill "$(cat "$PIDFILE")"
        echo "stopped"
    else
        echo "not running"
    fi
    rm -f "$PIDFILE"
}

case "${1:-start}" in
stop)
    stop_it
    exit 0
    ;;
logs)
    tail -f "$LOGFILE"
    exit 0
    ;;
start | fg) ;;
*)
    echo "usage: $0 [start|stop|logs|fg]" >&2
    exit 2
    ;;
esac

if [[ ! -d "$HERE/node_modules" ]]; then
    echo "installing just-bash…"
    (cd "$HERE" && npm install --silent)
fi

# Checked here because the failure otherwise arrives as an EADDRINUSE stack
# trace that never names the port's occupant — and on this project the likely
# occupant is one of our own services.
if command -v ss >/dev/null && ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
    echo "Port $PORT is already in use. Set BASH_PORT to something else:" >&2
    echo "  BASH_PORT=8900 $0" >&2
    exit 1
fi

if [[ "${1:-start}" == "fg" ]]; then
    exec env BASH_PORT="$PORT" node "$HERE/server.mjs"
fi

BASH_PORT="$PORT" nohup node "$HERE/server.mjs" >"$LOGFILE" 2>&1 &
echo $! >"$PIDFILE"

for _ in $(seq 20); do
    if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "ready — set BASH_URL=http://localhost:$PORT in backend/.env"
        exit 0
    fi
    sleep 0.5
done

echo "Started, but /health did not answer in 10s. Check:" >&2
echo "  tail $LOGFILE" >&2
stop_it >/dev/null || true
exit 1
