import pandas as pd
import os


def pull_roster(season: int) -> pd.DataFrame:
    url = f"https://www.sports-reference.com/cbb/schools/syracuse/{season}.html"
    tables = pd.read_html(url)

    df = tables[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)

    df = df[df["Player"] != "Team Totals"].copy()
    df["Season"] = season

    return df


def main():
    seasons = [2021, 2022, 2023, 2024, 2025]

    os.makedirs("data", exist_ok=True)

    all_data = []

    for season in seasons:
        print(f"Pulling {season} season...")
        df = pull_roster(season)
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    output_path = "data/syracuse_all_player_stats.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nSaved combined dataset to: {output_path}")
    print(f"Total rows: {len(combined)}")


if __name__ == "__main__":
    main()


