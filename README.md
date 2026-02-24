# Basketball Lineup Analytics Engine

A matchup-aware lineup optimization and rotation decision-support tool built for basketball analytics.

This application evaluates every viable 5-man lineup combination using advanced player metrics, contextual game-state adjustments, and a supervised machine learning model to project net rating impact.

Designed as a sports analytics / basketball operations tool.

## Core Capabilities
**1. Lineup Optimization**
Ranks all valid 5-man combinations from a configurable player pool based on:

- Predicted Net Rating (ML model)
- Offensive and Defensive Efficiency
- Ball Security
- Rebounding Impact
-Reliability Penalty (low-minute shrinkage)
- Contextual Objective Weighting
Constraints supported:
- Minimum guards
- Minimum bigs
- Maximum centers

**2. Game-State Simulation**
Adjust lineup rankings dynamically based on:

- Score margin
- Minutes remaining
- Strategic priority (Balanced, Need a Bucket, Get Stops, etc.)
- Fatigue level & fatigue impact strength
- Objective weights automatically adjust based on context.

**3. Matchup Modeling vs Opponent 5-Man Units**
Evaluates Syracuse lineups against specific ACC opponent lineups:

- Auto (Top 5 by minutes)
- Archetype (Big / Small / Shooting / Defense)
- Manual selection
- Includes a Level-1 matchup adjustment modeling:
- Defensive edge vs opponent ORtg
- Turnover risk vs opponent pressure

**4. Substitution Assistant**
Given a current 5-man lineup, the tool:

- Simulates all valid 1-for-1 substitutions
- Scores each swap under current objective
- Returns ranked substitution recommendations
- Rebounding differential

## Modeling Approach

- Player-level season metrics aggregated into lineup-level features
- Supervised regression model predicts projected net rating
- Reliability penalty applied for low-minute units
- Fatigue multiplier adjusts impact late-game
- Chemistry & familiarity multipliers stabilize predictions
- Matchup adjustment estimates opponent interaction effects
- Final Objective score = weighted normalized composite

This system is built as a decision-support engine — not a deterministic predictor.

## Tech Stack

- Python
- Streamlit
- Pandas / NumPy
- Scikit-Learn
- Joblib

## Data inputs
- `data/syracuse_player_profiles.csv` (Syracuse player season profiles)
- `data/acc_player_profiles.csv` (ACC player season profiles)
- `data/lineup_model.joblib` (trained model used to predict lineup net rating)

## Run locally 
```bash
git clone https://github.com/jacobsmith0408-svg/basketball-lineup-analytics-engine.git
cd basketball-lineup-analytics-engine

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

## Project Structure

- app.py        → Streamlit interface + scoring engine
- src/          → data processing / model scripts
- data/         → input CSVs + trained model artifact
- notebooks/    → model development experiments

## Limitations

- Model performance depends on feature quality and training data
- Matchup logic is heuristic-based (Level-1 interaction layer)
- Designed as a prototype decision-support tool, not a production analytics platform
