import pandas as pd

INPUT = "data/syracuse_all_player_stats.csv"
OUTPUT = "data/syracuse_player_profiles.csv"


def main():
    df = pd.read_csv(INPUT)

    # Keep only relevant columns
    cols = [
        "Player",
        "Season",
        "MP_adv",     # minutes played
        "ORtg",
        "DRtg",
        "OBPM",
        "DBPM",
        "BPM",
        "TS%",
        "eFG%",
        "USG%",
        "AST%",
        "TOV%",
        "ORB%",
        "DRB%",
        "STL%",
        "BLK%"
    ]

    df = df[cols].copy()

    # Convert everything numeric except Player/Season
    for col in cols:
        if col not in ["Player", "Season"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove players with no minutes
    df = df[df["MP_adv"] > 0]

    df.to_csv(OUTPUT, index=False)

    print("Saved player profiles to:", OUTPUT)
    print(df.head())


if __name__ == "__main__":
    main()
