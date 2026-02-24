import pandas as pd

DATA = "data/syracuse_player_profiles.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
    df["MP_adv"] = pd.to_numeric(df["MP_adv"], errors="coerce")
    return df


def list_players(season: int) -> list[str]:
    df = load_data()
    names = (
        df[df["Season"] == season]["Player"]
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(names)


def evaluate_lineup(season: int, players: list[str]) -> dict:
    if len(players) != 5:
        raise ValueError("Lineup must contain exactly 5 players.")

    df = load_data()
    sub = df[(df["Season"] == season) & (df["Player"].isin(players))].copy()

    if len(sub) != 5:
        missing = sorted(set(players) - set(sub["Player"].tolist()))
        raise ValueError(f"Could not find these players for {season}: {missing}")

    weights = sub["MP_adv"].fillna(0)
    if weights.sum() == 0:
        # fallback: equal weights
        weights = pd.Series([1, 1, 1, 1, 1], index=sub.index)

    def wavg(col: str) -> float:
        return float((sub[col] * weights).sum() / weights.sum())

    ortg = wavg("ORtg")
    drtg = wavg("DRtg")
    bpm = wavg("BPM")

    return {
        "Expected_ORtg": round(ortg, 2),
        "Expected_DRtg": round(drtg, 2),
        "Expected_Net": round(ortg - drtg, 2),
        "Expected_BPM": round(bpm, 2),
    }


if __name__ == "__main__":
    season = 2026
    players = list_players(season)
    print(f"Syracuse players in {season}: {len(players)}")
    print(players)
