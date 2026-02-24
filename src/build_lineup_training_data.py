import itertools
import numpy as np
import pandas as pd

PROFILES_PATH = "data/syracuse_player_profiles.csv"
OUTPUT_PATH = "data/syracuse_lineup_training.csv"

# --- Controls ---
SEASONS = [2021, 2022, 2023, 2024, 2025,2026]
TOP_N_BY_MINUTES = 12          # player pool per season for training
MIN_PLAYER_MINUTES = 100       # drop small players from the pool
N_LINEUPS_PER_SEASON = 1500    # how many lineups to sample per season
RANDOM_SEED = 42


def load_profiles() -> pd.DataFrame:
    df = pd.read_csv(PROFILES_PATH)
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce")

    numeric_cols = [
        "MP_adv", "ORtg", "DRtg", "BPM",
        "TS%", "USG%", "AST%", "TOV%", "ORB%", "DRB%", "STL%", "BLK%"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep rows with minutes
    df = df[df["MP_adv"].fillna(0) > 0].copy()

    return df


def weighted_sample_without_replacement(players: np.ndarray, weights: np.ndarray, k: int, rng) -> list[str]:
    """
    Sample k unique players without replacement, with probability proportional to weights.
    """
    weights = np.maximum(weights, 0).astype(float)
    if weights.sum() == 0:
        weights = np.ones_like(weights, dtype=float)

    probs = weights / weights.sum()
    chosen = rng.choice(players, size=k, replace=False, p=probs)
    return chosen.tolist()


def lineup_features(lineup_df: pd.DataFrame) -> dict:
    w = lineup_df["MP_adv"].fillna(0).astype(float)
    if w.sum() == 0:
        w = pd.Series([1.0] * len(lineup_df), index=lineup_df.index)

    def wavg(col: str) -> float:
        return float((lineup_df[col].astype(float) * w).sum() / w.sum())

    def wstd(col: str) -> float:
        x = lineup_df[col].astype(float)
        w_ = w / w.sum()
        mu = float((x * w_).sum())
        var = float((w_ * (x - mu) ** 2).sum())
        return var ** 0.5

    ortg = wavg("ORtg")
    drtg = wavg("DRtg")

    feats = {
        "ORtg_lineup": ortg,
        "DRtg_lineup": drtg,
        "Net_lineup": ortg - drtg,

        "BPM_lineup": wavg("BPM"),
        "TS_lineup": wavg("TS%"),
        "USG_lineup": wavg("USG%"),
        "USG_std_lineup": wstd("USG%"),

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



def build_training_for_season(df: pd.DataFrame, season: int, rng) -> pd.DataFrame:
    df_s = df[df["Season"] == season].copy()
    if df_s.empty:
        return pd.DataFrame()

    # filter by minutes
    df_s = df_s[df_s["MP_adv"] >= MIN_PLAYER_MINUTES].copy()
    if df_s.empty or len(df_s) < 5:
        return pd.DataFrame()

    # restrict pool to top N by minutes
    df_s = df_s.sort_values("MP_adv", ascending=False).head(TOP_N_BY_MINUTES).copy()
    if len(df_s) < 5:
        return pd.DataFrame()

    players = df_s["Player"].to_numpy()
    weights = df_s["MP_adv"].to_numpy(dtype=float)

    # Pre-index for fast selection
    df_idx = df_s.set_index("Player")

    seen = set()
    rows = []

    attempts = 0
    max_attempts = N_LINEUPS_PER_SEASON * 10  # avoid infinite loops if pool small

    while len(rows) < N_LINEUPS_PER_SEASON and attempts < max_attempts:
        attempts += 1

        lineup = weighted_sample_without_replacement(players, weights, k=5, rng=rng)
        lineup_key = tuple(sorted(lineup))

        if lineup_key in seen:
            continue
        seen.add(lineup_key)

        lineup_df = df_idx.loc[list(lineup)].reset_index()

        feats = lineup_features(lineup_df)
        feats["Season"] = int(season)
        feats["Players"] = " | ".join(lineup_key)

        rows.append(feats)

    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    df = load_profiles()

    all_rows = []
    for season in SEASONS:
        season_df = build_training_for_season(df, season, rng)
        print(f"Season {season}: generated {len(season_df)} unique lineups")
        all_rows.append(season_df)

    out = pd.concat(all_rows, ignore_index=True)

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved training data -> {OUTPUT_PATH}")
    print(out.head())


if __name__ == "__main__":
    main()

