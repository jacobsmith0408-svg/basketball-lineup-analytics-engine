import pandas as pd
import joblib

PROFILES_PATH = "data/syracuse_player_profiles.csv"
MODEL_PATH = "data/lineup_model.joblib"

FEATURE_COLUMNS = [
    "TS_lineup","USG_lineup","AST_lineup","TOV_lineup","ORB_lineup","DRB_lineup","STL_lineup","BLK_lineup",
    "BPM_lineup","ORtg_lineup","DRtg_lineup","MIN_total","MIN_avg"
]


def weighted_features(lineup_df: pd.DataFrame) -> dict:
    w = lineup_df["MP_adv"].fillna(0).astype(float)
    if w.sum() <= 0:
        w = pd.Series([1.0] * len(lineup_df), index=lineup_df.index)

    def wavg(col: str) -> float:
        return float((lineup_df[col] * w).sum() / w.sum())

    ortg = wavg("ORtg")
    drtg = wavg("DRtg")

    return {
        "TS_lineup": wavg("TS%"),
        "USG_lineup": wavg("USG%"),
        "AST_lineup": wavg("AST%"),
        "TOV_lineup": wavg("TOV%"),
        "ORB_lineup": wavg("ORB%"),
        "DRB_lineup": wavg("DRB%"),
        "STL_lineup": wavg("STL%"),
        "BLK_lineup": wavg("BLK%"),
        "BPM_lineup": wavg("BPM"),
        "ORtg_lineup": ortg,
        "DRtg_lineup": drtg,
        "MIN_total": float(lineup_df["MP_adv"].sum()),
        "MIN_avg": float(lineup_df["MP_adv"].mean()),
    }


def main():
    season = 2024
    df = pd.read_csv(PROFILES_PATH)
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
    df = df[df["Season"] == season].copy()

    model = joblib.load(MODEL_PATH)

    lineups = [
        ["JJ Starling", "Chris Bell", "Judah Mintz", "Maliq Brown", "Naheem McLeod"],
        ["Maliq Brown", "Quadir Copeland", "Kyle Cuffe Jr.", "Benny Williams", "Naheem McLeod"],
    ]

    rows = []
    for i, lineup in enumerate(lineups, start=1):
        sub = df[df["Player"].isin(lineup)].copy()
        if len(sub) != 5:
            missing = sorted(set(lineup) - set(sub["Player"].tolist()))
            raise ValueError(f"Lineup {i} missing players for season {season}: {missing}")

        feats = weighted_features(sub)
        X = pd.DataFrame([feats], columns=FEATURE_COLUMNS)
        pred = float(model.predict(X)[0])

        row = {"Lineup": i, "Players": " | ".join(lineup), "Predicted_Net": round(pred, 2)}
        for k, v in feats.items():
            row[k] = round(v, 3)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("Predicted_Net", ascending=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
