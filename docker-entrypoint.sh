#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] starting Xvfb..."
/usr/bin/Xvfb :99 -screen 0 1280x1024x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!
export DISPLAY=:99
echo "[entrypoint] DISPLAY=$DISPLAY (Xvfb PID=$XVFB_PID)"

echo "[entrypoint] launching webapp_server.py..."
exec python -u webapp_server.py
