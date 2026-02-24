# Basketball Lineup Analytics Engine

A matchup-aware lineup optimization and rotation decision-support tool built for basketball analytics.

This application evaluates every viable 5-man lineup combination using advanced player metrics, contextual game-state adjustments, and a supervised machine learning model to project net rating impact.

Designed as a sports analytics / basketball operations tool.

👉 **Live Demo:** (https://basketball-lineup-analytics-engine.streamlit.app/)

## What this does
**Lineup Recommendations**
- Ranks all 5-man combinations from a minutes-based player pool
- Supports roster constraints (min guards, min bigs, max centers)
- Optimizes for game-state priorities (Balanced / Need a bucket / Get stops / Protect the ball / Win the glass)

**Substitution Assistant**
- Given the current 5 + bench, suggests the best **one-for-one** substitutions
- Uses the same objective + constraints as the lineup ranking engine

**Matchups vs ACC Opponents**
- Scores Syracuse 5-man counters vs a specific opponent 5-man lineup
- Opponent lineup modes: auto (top minutes), archetype (Big/Small/Shooting/Defense), manual

## Screenshots
**Lineup Recommendations**
<img width="1440" height="777" alt="Screenshot 2026-02-24 at 2 18 23 PM" src="https://github.com/user-attachments/assets/4f81e5a2-6dc0-4116-9d39-7be0d5f95ec2" />

**Substitution Assistant**
<img width="1440" height="777" alt="Screenshot 2026-02-24 at 2 25 46 PM" src="https://github.com/user-attachments/assets/c9e92647-e5de-4b62-b915-604bedbeb6ad" />

**Matchup vs Opponent**

<img width="1440" height="775" alt="Screenshot 2026-02-24 at 2 17 23 PM" src="https://github.com/user-attachments/assets/9c177be7-e84b-4d8e-81a8-fc5083f1510a" />

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

## Limitations

- Model performance depends on feature quality and training data
- Matchup logic is heuristic-based (Level-1 interaction layer)
- Designed as a prototype decision-support tool, not a production analytics platform
