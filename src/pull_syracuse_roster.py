import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment


def _read_table_by_id(html: str, table_id: str) -> pd.DataFrame:
    """
    Sports-Reference often wraps tables in HTML comments.
    This function finds a table by id in both normal HTML and commented HTML.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) Try normal HTML first
    table = soup.find("table", id=table_id)
    if table is not None:
        return pd.read_html(str(table))[0]

    # 2) Try commented HTML
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for c in comments:
        if table_id in c:
            csoup = BeautifulSoup(c, "html.parser")
            table = csoup.find("table", id=table_id)
            if table is not None:
                return pd.read_html(str(table))[0]

    raise ValueError(f"Could not find table id='{table_id}'")


def pull_syracuse_season(season: int) -> pd.DataFrame:
    url = f"https://www.sports-reference.com/cbb/schools/syracuse/{season}.html"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # Per-game + Advanced are what we want
    per_game = _read_table_by_id(resp.text, "per_game")
    advanced = _read_table_by_id(resp.text, "advanced")

    # Clean: remove Team Totals
    per_game = per_game[per_game["Player"] != "Team Totals"].copy()
    advanced = advanced[advanced["Player"] != "Team Totals"].copy()

    # Merge on Player
    df = per_game.merge(advanced, on="Player", how="left", suffixes=("_per_game", "_adv"))
    df["Season"] = season
    return df


def main():
    seasons = [2021, 2022, 2023, 2024, 2025, 2026]
    all_rows = []

    for s in seasons:
        print(f"Pulling Syracuse {s}...")
        all_rows.append(pull_syracuse_season(s))

    combined = pd.concat(all_rows, ignore_index=True)

    # Save combined stats table
    combined.to_csv("data/syracuse_all_player_stats.csv", index=False)
    print("\nSaved: data/syracuse_all_player_stats.csv")
    print(combined.head())


if __name__ == "__main__":
    main()



