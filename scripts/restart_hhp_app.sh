#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_SESSION="${HHP_API_SESSION:-hhp_api}"
NGROK_SESSION="${HHP_NGROK_SESSION:-hhp_ngrok}"

for s in "$API_SESSION" "$NGROK_SESSION"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux kill-session -t "$s"
  fi
done

tmux new-session -d -s "$API_SESSION"   "cd '$ROOT_DIR' && ./scripts/run_hhp_api.sh"
tmux new-session -d -s "$NGROK_SESSION" "cd '$ROOT_DIR' && ./scripts/run_hhp_ngrok.sh"

printf 'HHP tmux sessions started:\n'
printf '  %s  -> uvicorn on 127.0.0.1:8002\n' "$API_SESSION"
printf '  %s -> ngrok http with control API on 127.0.0.1:4041\n' "$NGROK_SESSION"
printf '\nPublic URL:\n'
# Give ngrok a moment to initialize its tunnel, then fetch the URL
for _ in 1 2 3 4 5 6; do
  if url=$("$ROOT_DIR/scripts/get_hhp_ngrok_url.sh" 2>/dev/null); then
    printf '  %s\n' "$url"; exit 0
  fi
  sleep 2
done
printf '  (tunnel not up yet — run ./scripts/get_hhp_ngrok_url.sh manually in a moment)\n'
