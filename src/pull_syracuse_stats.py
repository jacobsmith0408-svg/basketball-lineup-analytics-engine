import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment


def read_table_by_id(html: str, table_id: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table", id=table_id)
    if table is not None:
        return pd.read_html(str(table))[0]

    for c in soup.find_all(string=lambda x: isinstance(x, Comment)):
        if table_id in c:
            csoup = BeautifulSoup(c, "html.parser")
            table = csoup.find("table", id=table_id)
            if table is not None:
                return pd.read_html(str(table))[0]

    raise ValueError(f"Table '{table_id}' not found")


def pull_season(season: int) -> pd.DataFrame:
    url = f"https://www.sports-reference.com/cbb/schools/syracuse/{season}.html"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    per_game = read_table_by_id(r.text, "players_per_game")
    advanced = read_table_by_id(r.text, "players_advanced")
    per_poss = read_table_by_id(r.text, "players_per_poss")  # NEW

    # Drop totals rows
    per_game = per_game[per_game["Player"] != "Team Totals"].copy()
    advanced = advanced[advanced["Player"] != "Team Totals"].copy()
    per_poss = per_poss[per_poss["Player"] != "Team Totals"].copy()

    # Merge all three
    df = per_game.merge(advanced, on="Player", how="left", suffixes=("_pg", "_adv"))
    df = df.merge(per_poss, on="Player", how="left", suffixes=("", "_poss"))

    df["Season"] = season
    return df


def main():
    seasons = [2021, 2022, 2023, 2024, 2025, 2026]
    all_data = []

    for s in seasons:
        print(f"Pulling Syracuse season {s}...")
        all_data.append(pull_season(s))

    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv("data/syracuse_all_player_stats.csv", index=False)

    print("\nSaved: data/syracuse_all_player_stats.csv")
    print("Columns sample:", combined.columns[:20].tolist())
    print(combined.head())


if __name__ == "__main__":
    main()
