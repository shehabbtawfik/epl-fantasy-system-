**Fantasy Premier League Optimization & Recommendation System**

A production-ready Python package that provides comprehensive FPL optimization, recommendations, and analysis tools built on top of Phase 1's advanced ML predictions.

## 🚀 Features

### Core Optimization Engine
- **FPL Rules-Compliant Optimizer**: Integer Linear Programming using PuLP
- **Complete Rule Enforcement**: 15-man squad, ≤3 per club, valid formations, £100m budget
- **Multiple Strategies**: Balanced, Premium, Value, and Differential optimization
- **Captain Selection**: Automatic captain and vice-captain optimization

### Comprehensive Recommendations
- **Optimal Squad Generation**: Multiple strategy-based squads
- **Positional Watchlists**: Top 15 GK, 25 DEF, 25 MID, 20 FWD
- **Top 50 Overall Rankings**: Complete player rankings with photos
- **Differential Analysis**: Low ownership, high potential players
- **Budget Enablers**: Cheap options to free up funds
- **Captaincy Recommendations**: Premium and safe captain options

### Production Tools
- **CLI Interface**: Complete command-line tools using Typer
- **Streamlit Dashboard**: Interactive web interface
- **CSV Export**: All recommendations exportable to CSV
- **Validation System**: Complete FPL rules compliance checking
- **Comprehensive Testing**: Full pytest test suite

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m fpl_tool.cli version
```

## 🎯 Quick Start

### 1. Command Line Interface

```bash
# Build/update dataset
python -m fpl_tool.cli build-dataset --seasons LAST3 --current

# Generate projections
python -m fpl_tool.cli project --gw CURRENT --horizon 6

# Optimize squad
python -m fpl_tool.cli optimize --budget 100.0 --strategy balanced

# Complete recommendations
python -m fpl_tool.cli recommend-gw --gw CURRENT --export output/recs.csv

# Validate squad
python -m fpl_tool.cli validate-squad squad.csv
```

### 2. Python API

```python
from fpl_tool import FPLOptimizer, FPLRecommender, FPLValidator

# Optimize squad
optimizer = FPLOptimizer()
result = optimizer.optimize_squad(budget=100.0, strategy="balanced")

# Generate recommendations
recommender = FPLRecommender()
recommendations = recommender.generate_complete_recommendations(gameweek=1)

# Validate compliance
validator = FPLValidator()
is_valid, errors = validator.validate_squad(squad)
```

### 3. Streamlit Dashboard

```bash
streamlit run fpl_tool/app_streamlit.py
```

## 📊 Output Examples

### Optimal Squad (Balanced Strategy)
```
Formation: 1-4-4-2
Total Cost: £97.0m
Expected Points: 31.3
Captain: Virgil van Dijk

Starting XI:
GKP: David Raya (Arsenal) - £5.5m
DEF: Virgil van Dijk (Liverpool) - £6.0m
DEF: William Saliba (Arsenal) - £6.0m
...
```

### Top 50 Overall Rankings
| Rank | Player | Position | Team | Price | xPts | Ownership | Photo |
|------|--------|----------|------|-------|------|-----------|-------|
| 1 | Virgil van Dijk | DEF | LIV | £6.0m | 3.2 | 15.2% | 🖼️ |
| 2 | Mohamed Salah | MID | LIV | £13.0m | 3.1 | 45.8% | 🖼️ |
| ... | ... | ... | ... | ... | ... | ... | ... |

### Watchlists by Position
- **Goalkeepers (15)**: Top GK options with clean sheet probabilities
- **Defenders (25)**: Best defensive assets with attacking potential  
- **Midfielders (25)**: High-scoring midfield options across price ranges
- **Forwards (20)**: Premium and budget forward options

## 🏗️ Architecture

```
fpl_tool/
├── __init__.py          # Package initialization
├── optimizer.py         # ILP optimization engine
├── recommender.py       # Complete recommendation system
├── validator.py         # FPL rules compliance checker
├── cli.py              # Typer-based CLI interface
└── app_streamlit.py    # Interactive dashboard

tests/
├── test_optimizer.py   # Optimizer test suite
├── test_recommender.py # Recommender test suite
├── test_validator.py   # Validator test suite
└── test_cli.py        # CLI test suite
```

## 🔧 Configuration

### Data Sources
- **Predictions**: `/home/ubuntu/data/fpl_xpts_predictions_enhanced.csv`
- **Master Data**: `/home/ubuntu/data/fpl_master_2025-26.csv`
- **Output**: `/home/ubuntu/output/`

### Optimization Parameters
- **Budget**: £100.0m (configurable)
- **Max per club**: 3 players (FPL rule)
- **Squad size**: 15 players (2 GK, 5 DEF, 5 MID, 3 FWD)
- **Starting XI**: 11 players in valid formation

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suite
python -m pytest tests/test_optimizer.py -v

# Run with coverage
python -m pytest tests/ --cov=fpl_tool --cov-report=html
```

## 📈 Performance

- **Optimization Speed**: ~2-5 seconds for 687 players
- **Memory Usage**: ~50MB for complete dataset
- **Accuracy**: 100% FPL rules compliance
- **Scalability**: Handles 1000+ players efficiently

## 🎮 Demo

Run the complete demonstration:

```bash
python demo_fpl_tool.py
```

This will test all components and generate sample outputs.

## 📋 CLI Commands Reference

### `build-dataset`
Build and update FPL dataset
```bash
python -m fpl_tool.cli build-dataset [OPTIONS]
```

### `project`
Generate expected points projections
```bash
python -m fpl_tool.cli project [OPTIONS]
```

### `optimize`
Optimize FPL squad under constraints
```bash
python -m fpl_tool.cli optimize [OPTIONS]
```

### `recommend-gw`
Generate complete gameweek recommendations
```bash
python -m fpl_tool.cli recommend-gw [OPTIONS]
```

### `validate-squad`
Validate squad compliance with FPL rules
```bash
python -m fpl_tool.cli validate-squad SQUAD_FILE [OPTIONS]
```

## 🔍 Validation Rules

The system enforces all official FPL rules:

1. **Squad Structure**: Exactly 15 players (2 GK, 5 DEF, 5 MID, 3 FWD)
2. **Club Limits**: Maximum 3 players per club
3. **Budget Constraint**: Total cost ≤ £100.0m
4. **Starting XI**: Valid formation (1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD)
5. **Captaincy**: Captain and vice-captain in starting XI, different players
6. **Player Status**: Only available players (status = 'a')

## 🚀 Weekly Workflow

1. **Update Data**: `python -m fpl_tool.cli build-dataset --current`
2. **Generate Projections**: `python -m fpl_tool.cli project --gw CURRENT`
3. **Get Recommendations**: `python -m fpl_tool.cli recommend-gw --gw CURRENT`
4. **Analyze Options**: Use Streamlit dashboard for interactive analysis
5. **Validate Transfers**: `python -m fpl_tool.cli validate-squad new_squad.csv`

## 📊 Integration with Phase 1

Phase 2 builds directly on Phase 1's outputs:
- **ML Predictions**: Uses `expected_points_ensemble` from enhanced models
- **Feature Engineering**: Leverages all 25+ engineered features
- **Fixture Analysis**: Incorporates DGW/BGW detection
- **Minutes Modeling**: Uses predicted minutes for rotation risk
- **Team Form**: Includes team strength metrics

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python -m pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FPL API**: Official Fantasy Premier League API
- **PuLP**: Linear programming optimization
- **Streamlit**: Interactive web applications
- **Typer**: Modern CLI framework
- **Phase 1 Team**: Advanced ML modeling and data engineering

---

**Built with ❤️ for the FPL community**

For support, feature requests, or bug reports, please open an issue on GitHub.
