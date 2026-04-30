# FPL Tool Container (CLI + Streamlit)
# Multi-purpose image for CLI workflows, scheduled refreshers, and the Streamlit dashboard

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Default app mode: cli | streamlit | refresh
    APP_MODE=cli \
    # Data/model/output directories (mounted as volumes recommended)
    APP_HOME=/app \
    DATA_DIR=/app/data \
    MODELS_DIR=/app/models \
    OUTPUT_DIR=/app/output \
    REFRESH_INTERVAL_HOURS=6

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates bash tini \
    && rm -rf /var/lib/apt/lists/*

# Create app dirs and non-root user
RUN useradd -ms /bin/bash appuser && mkdir -p "$APP_HOME" "$DATA_DIR" "$MODELS_DIR" "$OUTPUT_DIR"
WORKDIR $APP_HOME

# Copy only requirements first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fpl_tool ./fpl_tool
COPY fetch_fpl.py fixture_analysis.py minutes_model.py team_form.py model_xpts.py model_xpts_enhanced.py ./
COPY data ./data
COPY models ./models
COPY output ./output

# Entry script to switch between modes
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && chown -R appuser:appuser $APP_HOME

USER appuser
EXPOSE 8501

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "fpl_tool.cli", "--help"]

HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import importlib; importlib.import_module('fpl_tool'); print('ok')" || exit 1
