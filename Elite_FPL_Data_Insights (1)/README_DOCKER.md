# FPL Tool – Containerized Deployment

This repository includes a production-ready Docker setup for the FPL data pipeline, ML projections, optimizer, CLI, and Streamlit dashboard.

## Files
- `Dockerfile` – Multi-purpose image (CLI / Streamlit / refresher)
- `entrypoint.sh` – Mode switcher (APP_MODE: cli | streamlit | refresh)
- `docker-compose.yml` – Services for CLI, dashboard, and periodic refresh
- `.dockerignore` – Excludes caches and large artifacts

## Build
```bash
docker build -t fpl-tool:latest .
```

## Volumes (Persistence)
The container expects to persist datasets, models, and outputs via volumes:
- `/app/data` ↔ `./data`
- `/app/models` ↔ `./models`
- `/app/output` ↔ `./output`

> The application uses environment variables to support both container (`/app/*`) and local development (`/home/ubuntu/*`) paths.

## Run – One-off CLI
Examples (match the project’s CLI):
```bash
# Help
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/output:/app/output \
  fpl-tool:latest python -m fpl_tool.cli --help

# Build dataset (last 3 + current)
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/output:/app/output \
  fpl-tool:latest python -m fpl_tool.cli build-dataset --seasons LAST3 --current

# Project xPts (current GW, 6-GW horizon)
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/output:/app/output \
  fpl-tool:latest python -m fpl_tool.cli project --gw CURRENT --horizon 6

# Optimize 15-man squad (budget £100.0m)
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/output:/app/output \
  fpl-tool:latest python -m fpl_tool.cli optimize --budget 100.0 --max-per-club 3 --gw CURRENT

# End-to-end weekly recommendations (export CSV)
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/output:/app/output \
  fpl-tool:latest python -m fpl_tool.cli recommend-gw --gw CURRENT --export /app/output/recs_gw.csv --print-images
```

Windows PowerShell (replace volume flags accordingly):
```powershell
docker run --rm -v ${PWD}\data:/app/data -v ${PWD}\models:/app/models -v ${PWD}\output:/app/output `
  fpl-tool:latest python -m fpl_tool.cli --help
```

## Run – Streamlit Dashboard
```bash
# Using compose (recommended)
docker compose up -d fpl_streamlit
# Open http://localhost:8501

# Or with docker run
docker run --rm -p 8501:8501 \
  -e APP_MODE=streamlit \
  -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/output:/app/output \
  fpl-tool:latest
```

## Automated Refresh Service
A lightweight refresher service that rebuilds the dataset and recommendations on a schedule.
```bash
# Default interval: 6 hours (configurable via REFRESH_INTERVAL_HOURS)
docker compose up -d fpl_refresh
```
The refresher runs:
1) build-dataset (LAST3 + current)
2) project (CURRENT, 6-GW horizon)
3) recommend-gw (CSV to /app/output/recommendations_current.csv)

## Environment Variables
- `APP_MODE`: cli | streamlit | refresh (default: cli)
- `DATA_DIR`, `MODELS_DIR`, `OUTPUT_DIR`: defaults `/app/*`
- `REFRESH_INTERVAL_HOURS`: default 6

## Healthcheck
The image includes a simple Python import health check. For Streamlit, rely on compose’s restart policy if needed.

## Tests Inside Container
```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/output:/app/output \
  fpl-tool:latest pytest -q
```

## Notes
- Runs as non-root user with `tini` as PID 1 for signal handling.
- Internet access is required to fetch FPL endpoints at runtime.
- If you add `ortools` to `requirements.txt`, the image size will increase substantially; current setup uses `pulp` by default.
