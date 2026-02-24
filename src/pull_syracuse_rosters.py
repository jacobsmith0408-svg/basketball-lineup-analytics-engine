import pandas as pd

BASE = "https://www.sports-reference.com/cbb/schools/syracuse/{season}.html"
SEASONS = [2024, 2025, 2026]

def ht_to_inches(ht):
    if pd.isna(ht):
        return None
    s = str(ht).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        try:
            return int(a) * 12 + int(b)
        except:
            return None
    return None

def pull_roster(season: int) -> pd.DataFrame:
    url = BASE.format(season=season)
    tables = pd.read_html(url)

    roster = None
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if "Player" in cols and any(c in cols for c in ["Pos", "Pos.", "Position"]) and any(c in cols for c in ["Ht", "Height"]):
            roster = t.copy()
            roster.columns = cols
            break

    if roster is None:
        raise ValueError(f"Roster table not found for {season}")

    if "Pos" not in roster.columns:
        if "Pos." in roster.columns:
            roster = roster.rename(columns={"Pos.": "Pos"})
        elif "Position" in roster.columns:
            roster = roster.rename(columns={"Position": "Pos"})

    if "Ht" not in roster.columns and "Height" in roster.columns:
        roster = roster.rename(columns={"Height": "Ht"})

    roster["Season"] = season
    roster = roster[["Player", "Pos", "Ht", "Season"]].copy()

    roster["Player"] = roster["Player"].astype(str).str.replace("*", "", regex=False).str.strip()
    roster["Pos"] = roster["Pos"].astype(str).str.strip()
    roster["Ht_in"] = roster["Ht"].apply(ht_to_inches)

    roster = roster.drop(columns=["Ht"])

    return roster

def main():
    out = []
    for season in SEASONS:
        print(f"Pulling {season} roster...")
        out.append(pull_roster(season))

    df = pd.concat(out, ignore_index=True)
    df.to_csv("data/syracuse_all_players.csv", index=False)
    print("Saved data/syracuse_all_players.csv")
    print(df.head())

if __name__ == "__main__":
    main()


