import pandas as pd

STATS_FILE = "data/syracuse_all_player_stats.csv"
ROSTER_FILE = "data/syracuse_all_players.csv"
OUTPUT_FILE = "data/syracuse_player_profiles.csv"


def clean_player_name(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("*", "").strip()
    s = s.replace("\u00a0", " ")
    s = " ".join(s.split())
    return s


def normalize_pos(raw) -> str:
    if pd.isna(raw):
        return ""
    s = str(raw).strip().upper()
    # Priority: C > G > F
    if "C" in s:
        return "C"
    if "G" in s:
        return "G"
    if "F" in s:
        return "F"
    return ""


def detect_pos_col(df: pd.DataFrame) -> str:
    for c in ["Pos", "Pos.", "Position", "POS", "POS."]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "pos" in str(c).lower():
            return c
    raise ValueError(f"No position-like column found in roster. Columns: {list(df.columns)}")


def main():
    stats = pd.read_csv(STATS_FILE)
    roster = pd.read_csv(ROSTER_FILE)

    # seasons numeric
    stats["Season"] = pd.to_numeric(stats["Season"], errors="coerce")
    roster["Season"] = pd.to_numeric(roster["Season"], errors="coerce")

    # clean names
    stats["Player"] = stats["Player"].apply(clean_player_name)
    roster["Player"] = roster["Player"].apply(clean_player_name)

    # If roster uses Pos. / Position, normalize to Pos
    pos_col = detect_pos_col(roster)
    if pos_col != "Pos":
        roster = roster.rename(columns={pos_col: "Pos"}).copy()

    # If roster doesn't have Ht_in, keep going
    has_height = "Ht_in" in roster.columns

    roster["Pos"] = roster["Pos"].apply(normalize_pos)
    if has_height:
        roster["Ht_in"] = pd.to_numeric(roster["Ht_in"], errors="coerce")

    keep_cols = ["Player", "Season", "Pos"] + (["Ht_in"] if has_height else [])
    roster = roster[keep_cols].copy()

    merged = stats.merge(roster, on=["Player", "Season"], how="left")

    # Ensure Pos column exists
    if "Pos" not in merged.columns:
        merged["Pos"] = ""

    # Normalize any incoming pos strings
    merged["Pos"] = merged["Pos"].apply(normalize_pos)

    # ----- OPTIONAL ALIASES -----
    # ALIASES = {"William Kyle III": "William Kyle"}
    # merged["Player"] = merged["Player"].replace(ALIASES)
    # -----------------------------------------------------

    # Hard overrides for your tool (because SR position labels can be wrong for "true 5s")
    CENTER_OVERRIDES = {
        "William Kyle": "C",
        "William Kyle III": "C",
        "Ibrahim Souare": "C",
        "Ibrahim Souaré": "C",
    }
    merged["Pos"] = merged.apply(lambda r: CENTER_OVERRIDES.get(r["Player"], r["Pos"]), axis=1)

    # Fill missing positions using height when possible
    if has_height and "Ht_in" in merged.columns:
        merged["Ht_in"] = pd.to_numeric(merged["Ht_in"], errors="coerce")
        missing = merged["Pos"].isna() | (merged["Pos"].astype(str).str.strip() == "")

        # Heuristic buckets
        merged.loc[missing & (merged["Ht_in"] >= 82), "Pos"] = "C"   # 6'10"+ = big/5
        merged.loc[missing & (merged["Ht_in"] < 78), "Pos"] = "G"    # <6'6" = guard
        merged.loc[missing & (merged["Pos"].isna() | (merged["Pos"] == "")), "Pos"] = "F"  # otherwise wing/big

    # If still missing, default to F
    merged["Pos"] = merged["Pos"].fillna("")
    merged.loc[merged["Pos"] == "", "Pos"] = "F"

    # Numeric conversion for metrics used downstream
    numeric_cols = [
        "MP_adv", "ORtg", "DRtg", "OBPM", "DBPM", "BPM",
        "TS%", "eFG%", "USG%", "AST%", "TOV%", "ORB%", "DRB%", "STL%", "BLK%"
    ]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Require minutes
    if "MP_adv" not in merged.columns:
        raise ValueError(f"MP_adv not found in stats file. Stats columns: {list(stats.columns)}")
    merged = merged[merged["MP_adv"].fillna(0) > 0].copy()

    output_cols = [
        "Player", "Season", "Pos", "MP_adv",
        "ORtg", "DRtg", "OBPM", "DBPM", "BPM",
        "TS%", "eFG%", "USG%", "AST%", "TOV%",
        "ORB%", "DRB%", "STL%", "BLK%"
    ]
    if has_height:
        output_cols.insert(3, "Ht_in")  # nice to keep for debugging

    output_cols = [c for c in output_cols if c in merged.columns]
    profiles = merged[output_cols].copy()

    profiles.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved profiles -> {OUTPUT_FILE}")

    # Sanity check for 2026
    season = 2026
    top = profiles[profiles["Season"] == season].sort_values("MP_adv", ascending=False).head(12)
    print(f"\nTop 12 by minutes for Season {season}:")
    cols_to_show = ["Player", "Pos", "MP_adv"] + (["Ht_in"] if "Ht_in" in top.columns else [])
    print(top[cols_to_show].to_string(index=False))
    print("\nCounts:", top["Pos"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()











