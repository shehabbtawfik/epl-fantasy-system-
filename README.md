# ⚽ FPL Optimizer — Live Fantasy Premier League Dashboard

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B)](https://shehab-epl.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

> **A production-grade FPL tool that fetches live player data from the official Premier League API and runs an Integer Linear Programming optimizer to build the highest-scoring squad possible — fully compliant with all FPL rules.**

---

## 🚀 Live Demo

**[→ Open the live app](https://shehab-epl.streamlit.app)**

The demo pulls **real-time player data** directly from `fantasy.premierleague.com` — no CSV files, no manual updates. Every squad it builds is valid under FPL rules.

---

## What This Project Demonstrates

| Capability | What It Shows |
|---|---|
| **Integer Linear Programming** | PuLP/CBC solver building optimal squads from 700+ players under hard constraints |
| **Live API integration** | Real-time data from the official FPL API, cached and refreshed every 5 minutes |
| **Multiple optimization strategies** | Balanced, Premium, Value, Differential, Form — each tuning the objective function differently |
| **FPL rules enforcement** | 15-man squad, ≤3 per club, valid formations, £100m budget — all as LP constraints |
| **Data engineering pipeline** | ML-based expected-points modeling (xPts) built on 3 seasons of historical data |
| **Interactive dashboard** | Streamlit app with squad optimizer, player analysis, watchlists, and transfer trends |

---

## Features

### 🎯 Squad Optimizer
- ILP solver (PuLP + CBC) finds the mathematically optimal squad
- 5 strategies: **Balanced** (xPts), **Premium** (total points), **Value** (xPts/£), **Differential** (form), **Form** (recent)
- Configurable budget (£90–100m) and max players per club (1–3)
- Automatic captain & vice-captain selection
- CSV export of any generated squad

### 📊 Player Analysis
- Interactive scatter: Price vs xPts for all 700+ players
- Filter by position, price range, team
- Sort by any metric: xPts, form, total points, value score, ownership

### 📋 Positional Watchlists
- Top 15 GK, 25 DEF, 25 MID, 20 FWD ranked by expected points
- Downloadable CSVs for each position

### 🏆 Top 50 Rankings
- Overall rankings with bar chart visualization
- Filter by position, adjustable N

### 🔄 Transfer Trends
- Most transferred in/out this gameweek
- Net transfer gain chart to spot the herd moves

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/shehabbtawfik/epl-fantasy-system-.git
cd epl-fantasy-system-

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard (fetches live data automatically)
streamlit run streamlit_app.py
```

Open http://localhost:8501 — no API keys, no data files needed.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│           FPL Official API                  │
│   fantasy.premierleague.com/api/            │
│   bootstrap-static/ (700+ players, live)    │
└──────────────────┬──────────────────────────┘
                   │ requests + 5-min cache
┌──────────────────▼──────────────────────────┐
│           Data Layer (pandas)               │
│  Player stats · Prices · xPts · Form        │
│  Ownership · Transfer deltas · Fitness      │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│        ILP Optimizer (PuLP/CBC)             │
│  Objective: maximise Σ score(player)        │
│  Constraints:                               │
│   · Squad = 15 players                      │
│   · 2 GK, 5 DEF, 5 MID, 3 FWD              │
│   · ≤ 3 players per club                   │
│   · Total cost ≤ budget                     │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         Streamlit Dashboard                 │
│  Home · Optimizer · Analysis ·              │
│  Watchlists · Top 50 · Transfers            │
└─────────────────────────────────────────────┘
```

### ML Pipeline (Phase 1 — in `fpl_tool/`)

The `fpl_tool` package contains a full ML-based xPts engine used for offline analysis:

- **Ensemble model**: Random Forest + Gradient Boosting + Ridge regression
- **25+ features**: Form, fixture difficulty, home/away, minutes modelling, rotation risk, DGW/BGW detection
- **Historical data**: 3 seasons of FPL data with engineered features
- **Outputs**: `fpl_xpts_predictions_enhanced.csv` used by the `FPLRecommender` class

The live Streamlit app uses FPL's own `ep_next` field as the xPts signal (always current), while the ML pipeline provides deeper analysis when run locally with historical data.

---

## Project Structure

```
epl-fantasy-system-/
├── streamlit_app.py          # ✨ Main live dashboard (Streamlit Cloud ready)
├── requirements.txt          # Dependencies
├── .streamlit/
│   └── config.toml           # Dark theme + server config
│
├── fpl_tool/                 # ML-powered recommendation engine
│   ├── optimizer.py          # ILP optimization engine
│   ├── recommender.py        # Full recommendation system
│   ├── validator.py          # FPL rules compliance checker
│   ├── cli.py                # Typer CLI interface
│   └── app_streamlit.py      # Legacy Streamlit app (uses local CSV data)
│
├── data/                     # Sample data & analysis outputs
├── models/                   # Trained ML models (.pkl)
├── output/                   # Pre-computed squad recommendations
└── tests/                    # pytest test suite
```

---

## CLI Usage (with local ML data)

```bash
# Build/update dataset
python -m fpl_tool.cli build-dataset --seasons LAST3 --current

# Generate xPts projections
python -m fpl_tool.cli project --gw CURRENT --horizon 6

# Optimize squad from local predictions
python -m fpl_tool.cli optimize --budget 100.0 --strategy balanced

# Full gameweek recommendations
python -m fpl_tool.cli recommend-gw --gw CURRENT --export output/recs.csv

# Validate a squad CSV
python -m fpl_tool.cli validate-squad squad.csv
```

---

## Deploying to Streamlit Community Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select your fork → set **Main file path** to `streamlit_app.py`
4. Click Deploy — that's it. No secrets or env vars required.

---

## Running Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=fpl_tool --cov-report=html
```

---

## License

MIT License — Copyright 2025 shehabbtawfik

---

*Built for the FPL community — and as a showcase of applied ML, optimization, and data engineering.*
