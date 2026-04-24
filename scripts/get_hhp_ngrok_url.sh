#!/usr/bin/env bash
set -euo pipefail

API_URL="${NGROK_API_URL:-http://127.0.0.1:4041/api/tunnels}"

python3 - <<PY
import json, os, sys, urllib.request
api_url = os.getenv("NGROK_API_URL", "$API_URL")
try:
    with urllib.request.urlopen(api_url, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
except Exception as exc:
    raise SystemExit(f"Could not read ngrok tunnel API at {api_url}: {exc}")

tunnels = payload.get("tunnels", [])
for t in tunnels:
    url = t.get("public_url", "")
    if url.startswith("https://"):
        print(url); sys.exit(0)
for t in tunnels:
    url = t.get("public_url", "")
    if url:
        print(url); sys.exit(0)
raise SystemExit("No ngrok tunnel found.")
PY
