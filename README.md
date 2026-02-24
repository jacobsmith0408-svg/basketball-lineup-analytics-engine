# Syracuse Lineup Efficiency (Streamlit)

Interactive lineup-ranking + substitution assistant for Syracuse basketball.

## What it does
- **Lineup Recommendations**: ranks all 5-man combos from your minutes-based player pool with constraints (min guards/bigs, max centers) and game-state priorities.
- **Substitution Assistant**: suggests the best 1-for-1 sub given the current five, the bench, and the selected objective.
- **Matchups vs ACC Opponents**: evaluates Syracuse 5-man counters vs an opponent 5-man lineup (auto, archetype, or manual).

## Data inputs
- `data/syracuse_player_profiles.csv` (Syracuse player season profiles)
- `data/acc_player_profiles.csv` (ACC player season profiles)
- `data/lineup_model.joblib` (trained model used to predict lineup net rating)

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes / limitations
- Predictions depend on the model + the available player profile columns (ORtg/DRtg/BPM/etc.).
- Opponent matchup logic is a **heuristic adjustment** (Level-1) layered on top of the Syracuse-only model.
- This tool is intended as decision support, not an oracle.

## Project structure
- `app.py` — Streamlit UI + scoring orchestration
- `src/` — data pull / model scripts
- `data/` — csv inputs + trained model artifact
