import os
import pandas as pd

ROSTER_FILE = "data/acc_all_players.csv"
STATS_FILE = "data/acc_all_player_stats.csv"
OUT_FILE = "data/acc_player_profiles.csv"

ALIASES = {}  # optional name fixes


def normalize_pos(p: str) -> str:
    if pd.isna(p):
        return ""
    p = str(p).strip().upper()
    if "C" in p:
        return "C"
    if "G" in p:
        return "G"
    return "F"


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    if not os.path.exists(ROSTER_FILE):
        raise FileNotFoundError(f"Missing {ROSTER_FILE}. Run pull_acc_teams.py first.")
    if not os.path.exists(STATS_FILE):
        raise FileNotFoundError(f"Missing {STATS_FILE}. Run pull_acc_stats.py first.")

    roster = pd.read_csv(ROSTER_FILE)
    stats = pd.read_csv(STATS_FILE)

    roster["Player"] = roster["Player"].astype(str).replace(ALIASES).str.strip()
    stats["Player"] = stats["Player"].astype(str).replace(ALIASES).str.strip()

    # unify roster position column
    if "Pos" not in roster.columns and "Position" in roster.columns:
        roster = roster.rename(columns={"Position": "Pos"})

    merged = pd.merge(
        stats,
        roster[["Team", "Season", "Player", "Pos"]],
        on=["Team", "Season", "Player"],
        how="left",
    )

    merged["Pos"] = merged["Pos"].apply(normalize_pos)

    # ensure minutes column exists
    if "MP_adv" not in merged.columns:
        if "MP" in merged.columns:
            merged["MP_adv"] = merged["MP"]
        elif "MP_pg" in merged.columns:
            merged["MP_adv"] = merged["MP_pg"]
        else:
            merged["MP_adv"] = 0

    # robustly map advanced stats columns into canonical names
    col_map = {}

    ortg_col = first_existing(merged, ["ORtg", "ORtg_adv", "ORtg_x", "ORtg_y"])
    drtg_col = first_existing(merged, ["DRtg", "DRtg_adv", "DRtg_x", "DRtg_y"])
    bpm_col  = first_existing(merged, ["BPM", "BPM_adv", "BPM_x", "BPM_y"])

    ts_col   = first_existing(merged, ["TS%", "TS%_adv", "TS%_x", "TS%_y"])
    usg_col  = first_existing(merged, ["USG%", "USG%_adv", "USG%_x", "USG%_y"])
    ast_col  = first_existing(merged, ["AST%", "AST%_adv", "AST%_x", "AST%_y"])
    tov_col  = first_existing(merged, ["TOV%", "TOV%_adv", "TOV%_x", "TOV%_y"])
    orb_col  = first_existing(merged, ["ORB%", "ORB%_adv", "ORB%_x", "ORB%_y"])
    drb_col  = first_existing(merged, ["DRB%", "DRB%_adv", "DRB%_x", "DRB%_y"])
    stl_col  = first_existing(merged, ["STL%", "STL%_adv", "STL%_x", "STL%_y"])
    blk_col  = first_existing(merged, ["BLK%", "BLK%_adv", "BLK%_x", "BLK%_y"])

    # create canonical columns
    def add_canon(canon: str, src: str | None):
        if src is None:
            merged[canon] = 0.0
        else:
            merged[canon] = merged[src]

    add_canon("ORtg", ortg_col)
    add_canon("DRtg", drtg_col)
    add_canon("BPM",  bpm_col)

    add_canon("TS%", ts_col)
    add_canon("USG%", usg_col)
    add_canon("AST%", ast_col)
    add_canon("TOV%", tov_col)
    add_canon("ORB%", orb_col)
    add_canon("DRB%", drb_col)
    add_canon("STL%", stl_col)
    add_canon("BLK%", blk_col)

    keep = [
        "Team", "Season", "Player", "Pos", "MP_adv",
        "ORtg", "DRtg", "BPM",
        "TS%", "USG%", "AST%", "TOV%",
        "ORB%", "DRB%", "STL%", "BLK%",
    ]

    out = merged[keep].copy()

    # numeric coercion
    for c in keep:
        if c in ["Team", "Player", "Pos"]:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out.to_csv(OUT_FILE, index=False)
    print(f"Saved: {OUT_FILE}")
    print("Example cols:", out.columns.tolist())
    print("Nonzero ORtg rows:", int((out["ORtg"] != 0).sum()))
    print("Nonzero DRtg rows:", int((out["DRtg"] != 0).sum()))


if __name__ == "__main__":
    main()