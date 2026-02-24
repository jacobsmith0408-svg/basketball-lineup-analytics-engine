import os
import pandas as pd

TEAM_SLUGS = {
    "Syracuse": "syracuse",
    "Duke": "duke",
    "North Carolina": "north-carolina",
    "Miami (FL)": "miami-fl",
    "Virginia": "virginia",
    "Clemson": "clemson",
}

SEASONS = [2024, 2025, 2026]
OUT_DIR = "data"


def pull_team_roster(team_slug: str, season: int) -> pd.DataFrame:
    url = f"https://www.sports-reference.com/cbb/schools/{team_slug}/men/{season}.html"
    tables = pd.read_html(url)

    # same roster table I used before
    roster = None
    for t in tables:
        if "Player" in t.columns and ("Pos" in t.columns or "Position" in t.columns):
            # heuristic: roster table usually has "No." / "Ht" / "Wt" / "Class"
            if any(c in t.columns for c in ["No.", "Ht", "Wt", "Class"]):
                roster = t.copy()
                break

    if roster is None:
        raise ValueError(f"Roster table not found for {team_slug} {season}")

    roster["Player"] = roster["Player"].astype(str).str.replace("*", "", regex=False).str.strip()
    roster["Season"] = season
    return roster


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []

    for team_name, slug in TEAM_SLUGS.items():
        for season in SEASONS:
            print(f"Pulling roster: {team_name} ({slug}) {season}...")
            df = pull_team_roster(slug, season)
            df["Team"] = team_name
            all_rows.append(df)

    out = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(OUT_DIR, "acc_all_players.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()