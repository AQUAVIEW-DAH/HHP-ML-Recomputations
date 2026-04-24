#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/hhp-env/bin/python}"
PORT="${HHP_API_PORT:-8002}"
LOG_FILE="${HHP_API_LOG:-$ROOT_DIR/logs/api_hhp_$PORT.log}"
MODEL_PATH="${HHP_MODEL_PATH:-$ROOT_DIR/artifacts/models/argo_gdac_teos10_2024_allstorms_rf_delta_model.pkl}"

cd "$ROOT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

APP_MODE=milton_replay HHP_MODEL_PATH="$MODEL_PATH" "$PYTHON_BIN" -m uvicorn api:app \
  --host 127.0.0.1 --port "$PORT" > "$LOG_FILE" 2>&1
