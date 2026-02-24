import itertools
import joblib
import pandas as pd


# -----------------------------
# CONFIG
# -----------------------------
SEASON = 2026

PROFILES_CSV = "data/syracuse_player_profiles.csv"
MODEL_PATH = "data/lineup_model.joblib"
OUT_CSV = f"data/ranked_lineups_{SEASON}_constrained.csv"

TOP_N_BY_MINUTES = 10
MIN_PLAYER_MINUTES = 250  # hard floor to enter player pool

# Constraints
MIN_GUARDS = 2
MIN_BIGS = 2          # BIG = F or C
MAX_CENTERS = 1

# Reliability penalty
MIN_MINUTES_SOFT = 500   # if the weakest-link player is below this, lineup gets penalized
PENALTY_SCALE = 3.0      # bigger = harsher penalty

# Optional hard excludes
EXCLUDE_PLAYERS = {
    # "Bryce Zephir",
}

# -----------------------------
# Helpers
# -----------------------------
def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nHave: {list(df.columns)}")


def build_combo_features(lineup_df: pd.DataFrame) -> dict:
    """
    Builds the EXACT feature schema used in training.
    Assumes build_lineup_training_data produced features using weighted averages of player stats.
    """
    w = lineup_df["MP_adv"].fillna(0).astype(float)
    if w.sum() == 0:
        w = pd.Series([1.0] * len(lineup_df), index=lineup_df.index)

    def wavg(col: str) -> float:
        return float((lineup_df[col].astype(float) * w).sum() / w.sum())

    # core
    ortg = wavg("ORtg")
    drtg = wavg("DRtg")

    feats = {
        "ORtg_lineup": ortg,
        "DRtg_lineup": drtg,
        "BPM_lineup": wavg("BPM"),
        "TS_lineup": wavg("TS%"),
        "USG_lineup": wavg("USG%"),
        "AST_lineup": wavg("AST%"),
        "TOV_lineup": wavg("TOV%"),
        "ORB_lineup": wavg("ORB%"),
        "DRB_lineup": wavg("DRB%"),
        "STL_lineup": wavg("STL%"),
        "BLK_lineup": wavg("BLK%"),
        "MIN_total": float(lineup_df["MP_adv"].sum()),
        "MIN_avg": float(lineup_df["MP_adv"].mean()),
    }
    return feats


def pos_counts(lineup_df: pd.DataFrame) -> dict:
    pos = lineup_df["Pos"].fillna("").astype(str).str.strip().str.upper()
    g = int((pos == "G").sum())
    f = int((pos == "F").sum())
    c = int((pos == "C").sum())
    big = int(((pos == "F") | (pos == "C")).sum())
    return {"G": g, "F": f, "C": c, "BIG": big}


def passes_constraints(counts: dict) -> bool:
    if counts["G"] < MIN_GUARDS:
        return False
    if counts["BIG"] < MIN_BIGS:
        return False
    if counts["C"] > MAX_CENTERS:
        return False
    return True


def reliability_penalty(lineup_df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Two-part penalty:
      1) weakest-link penalty (min minutes)
      2) depth penalty (how many players are below MIN_MINUTES_SOFT)
    Returns: (penalty_total, min_mp, avg_mp)
    """
    mps = lineup_df["MP_adv"].astype(float).tolist()
    min_mp = float(min(mps))
    avg_mp = float(sum(mps) / len(mps))

    # Part 1: weakest link
    gap_min = max(0.0, (MIN_MINUTES_SOFT - min_mp) / MIN_MINUTES_SOFT)
    p1 = PENALTY_SCALE * gap_min

    # Part 2: depth penalty (penalize multiple sub-threshold guys)
    gaps = [max(0.0, (MIN_MINUTES_SOFT - mp) / MIN_MINUTES_SOFT) for mp in mps]
    p2 = (PENALTY_SCALE * 0.6) * (sum(gaps) / len(gaps))

    return float(p1 + p2), min_mp, avg_mp


# -----------------------------
# Main
# -----------------------------
def main():
    df = pd.read_csv(PROFILES_CSV)

    # Basic cleaning / numeric coercion
    ensure_columns(df, ["Player", "Season", "Pos", "MP_adv", "ORtg", "DRtg", "BPM", "TS%", "USG%", "AST%", "TOV%", "ORB%", "DRB%", "STL%", "BLK%"])
    df["Season"] = safe_num(df["Season"])
    df["MP_adv"] = safe_num(df["MP_adv"])

    season_df = df[df["Season"] == SEASON].copy()

    if season_df.empty:
        raise ValueError(f"No rows found for Season={SEASON} in {PROFILES_CSV}")

    # Excludes
    if EXCLUDE_PLAYERS:
        season_df = season_df[~season_df["Player"].isin(EXCLUDE_PLAYERS)].copy()

    # Hard minutes floor just for the pool
    pool_df = season_df[season_df["MP_adv"].fillna(0) >= MIN_PLAYER_MINUTES].copy()

    # Top-N by minutes pool
    pool_df = pool_df.sort_values("MP_adv", ascending=False).head(TOP_N_BY_MINUTES).copy()

    players = pool_df["Player"].tolist()

    if len(players) < 5:
        raise ValueError(f"Not enough players in pool to form lineups. Pool size={len(players)}. "
                         f"Lower MIN_PLAYER_MINUTES or raise TOP_N_BY_MINUTES.")

    print(f"\nSeason: {SEASON} | Player pool: {len(players)} | "
          f"Combos possible: {len(list(itertools.combinations(players, 5)))}")
    print("\nPlayer pool (by minutes):")
    print(pool_df[["Player", "Pos", "MP_adv"]].to_string(index=False))

    # Load model
    model = joblib.load(MODEL_PATH)

    # IMPORTANT: feature order must match training
    model_features = None
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)

    results = []

    for combo in itertools.combinations(players, 5):
        lineup_df = season_df[season_df["Player"].isin(combo)].copy()
        if len(lineup_df) != 5:
            continue

        counts = pos_counts(lineup_df)
        if not passes_constraints(counts):
            continue

        feats = build_combo_features(lineup_df)

        # Build X
        X = pd.DataFrame([feats])

        # Align columns to model expectation
        if model_features is not None:
            for c in model_features:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[model_features]
        else:
            # If model doesn't store feature names, you must keep training/ranking in sync manually.
            # This fallback tries to be reasonable.
            X = X.reindex(sorted(X.columns), axis=1)

        pred_raw = float(model.predict(X)[0])

        pen, min_mp, avg_mp = reliability_penalty(lineup_df)

        pred_adj = pred_raw - pen

        results.append({
            "Players": " | ".join(combo),
            "Predicted_Net": pred_adj,
            "Predicted_Net_raw": pred_raw,
            "Penalty": pen,
            "Min_MP_in_lineup": min_mp,
            "Avg_MP_in_lineup": avg_mp,
            "ORtg_lineup": feats["ORtg_lineup"],
            "DRtg_lineup": feats["DRtg_lineup"],
            "MIN_total": feats["MIN_total"],
            "G": counts["G"],
            "F": counts["F"],
            "C": counts["C"],
        })

    if not results:
        print("\nNo lineups passed constraints.")
        print("Loosen constraints or expand pool:")
        print("- increase TOP_N_BY_MINUTES")
        print("- lower MIN_PLAYER_MINUTES")
        print("- lower MIN_GUARDS / MIN_BIGS or raise MAX_CENTERS")
        return

    ranked = pd.DataFrame(results).sort_values("Predicted_Net", ascending=False)

    print("\nTop 20 predicted lineups (constrained):\n")
    show_cols = ["Players", "Predicted_Net", "Predicted_Net_raw", "Penalty", "Min_MP_in_lineup", "Avg_MP_in_lineup",
                 "ORtg_lineup", "DRtg_lineup", "MIN_total", "G", "F", "C"]
    print(ranked[show_cols].head(20).to_string(index=False))

    ranked.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()





