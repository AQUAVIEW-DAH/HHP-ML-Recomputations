#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NGROK_BIN="${NGROK_BIN:-/home/suramya/MLD-ML-Recomputation/bin/ngrok}"
PORT="${HHP_PUBLIC_PORT:-8002}"
USER_CONFIG="${NGROK_USER_CONFIG:-$HOME/.config/ngrok/ngrok.yml}"
HHP_CONFIG="${HHP_NGROK_CONFIG:-$ROOT_DIR/deploy/ngrok/ngrok_hhp.yml}"
LOG_DIR="$ROOT_DIR/logs"

mkdir -p "$LOG_DIR"

if [[ ! -x "$NGROK_BIN" ]]; then
  echo "ngrok binary not found at $NGROK_BIN" >&2
  exit 1
fi

# Dynamic URL tunnel (non-static). Merge user config (auth token) with HHP
# config (custom web_addr on 4041 so we don't collide with MLD's 4040).
exec "$NGROK_BIN" http "http://127.0.0.1:${PORT}" \
  --config "$USER_CONFIG","$HHP_CONFIG" \
  --log stdout --log-level info
