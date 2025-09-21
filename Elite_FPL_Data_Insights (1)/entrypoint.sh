#!/usr/bin/env bash
set -euo pipefail

APP_MODE=${APP_MODE:-cli}
APP_HOME=${APP_HOME:-/app}
DATA_DIR=${DATA_DIR:-/app/data}
MODELS_DIR=${MODELS_DIR:-/app/models}
OUTPUT_DIR=${OUTPUT_DIR:-/app/output}
REFRESH_INTERVAL_HOURS=${REFRESH_INTERVAL_HOURS:-6}
REFRESH_INTERVAL_SECONDS=$((REFRESH_INTERVAL_HOURS*3600))

# Ensure app paths exist
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$OUTPUT_DIR"

case "$APP_MODE" in
  streamlit)
    echo "[entrypoint] Starting Streamlit dashboard on :8501"
    cd "$APP_HOME"
    export PYTHONPATH="$APP_HOME:${PYTHONPATH:-}"
    exec streamlit run "$APP_HOME/fpl_tool/app_streamlit.py" --server.port 8501 --server.address 0.0.0.0
    ;;
  refresh)
    echo "[entrypoint] Starting periodic refresh every ${REFRESH_INTERVAL_HOURS}h"
    while true; do
      date
      python -m fpl_tool.cli build-dataset --seasons LAST3 --current || true
      python -m fpl_tool.cli project --gw CURRENT --horizon 6 || true
      python -m fpl_tool.cli recommend-gw --gw CURRENT --export "$OUTPUT_DIR/recommendations_current.csv" || true
      echo "[entrypoint] Sleeping for ${REFRESH_INTERVAL_SECONDS}s..."
      sleep "$REFRESH_INTERVAL_SECONDS"
    done
    ;;
  cli|*)
    echo "[entrypoint] Running CLI: $*"
    exec "$@"
    ;;
esac
