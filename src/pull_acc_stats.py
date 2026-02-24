import os
import re
from io import StringIO

import pandas as pd
import requests

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
OUT_FILE = os.path.join(OUT_DIR, "acc_all_player_stats.csv")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}


def _extract_table_html(html: str, table_id: str) -> str:
    pattern = rf'(<table[^>]*id="{re.escape(table_id)}"[^>]*>.*?</table>)'
    m = re.search(pattern, html, flags=re.DOTALL)
    if m:
        return m.group(1)

    for block in re.findall(r"<!--(.*?)-->", html, flags=re.DOTALL):
        m2 = re.search(pattern, block, flags=re.DOTALL)
        if m2:
            return m2.group(1)

    return ""


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        flat = []
        for a, b in df.columns:
            name = b if (b and "Unnamed" not in str(b)) else a
            flat.append(str(name).strip())
        df.columns = flat
    else:
        df.columns = [str(c).strip() for c in df.columns]

    df.columns = [c.replace("\xa0", " ").strip() for c in df.columns]
    return df


def _drop_repeat_headers(df: pd.DataFrame) -> pd.DataFrame:
    if "Player" in df.columns:
        df = df[df["Player"].astype(str).str.strip().ne("Player")].copy()
    return df

def _drop_non_players(df: pd.DataFrame) -> pd.DataFrame:
    if "Player" not in df.columns:
        return df

    bad = {
        "Team Totals",
        "School Totals",
        "Totals",
        "Opponent Totals",
    }

    p = df["Player"].astype(str).str.strip()
    df = df[~p.isin(bad)].copy()
    df = df[~p.str.contains("Totals", case=False, na=False)].copy()  # catches weird variants
    return df

def _read_table(table_html: str) -> pd.DataFrame:
    df = pd.read_html(StringIO(table_html))[0]
    df = _clean_columns(df)
    df = _drop_repeat_headers(df)
    df = _drop_non_players(df)
    if "Player" in df.columns:
        df["Player"] = df["Player"].astype(str).str.replace("*", "", regex=False).str.strip()
    return df

def pull_team_stats(team_slug: str, season: int) -> pd.DataFrame:
    url = f"https://www.sports-reference.com/cbb/schools/{team_slug}/men/{season}.html"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    html = r.text

    # tables I need
    pg_html = _extract_table_html(html, "players_per_game")
    adv_html = _extract_table_html(html, "players_advanced")
    poss_html = _extract_table_html(html, "players_per_poss")

    if not pg_html or not adv_html or not poss_html:
        ids = sorted(set(re.findall(r'id="([^"]+)"', html)))
        raise ValueError(
            f"Missing required tables for {team_slug} {season}. "
            f"Need: players_per_game, players_advanced, players_per_poss. "
            f"IDs sample: {ids[:50]}"
        )

    pg = _read_table(pg_html)
    adv = _read_table(adv_html)
    poss = _read_table(poss_html)

    # add Season for merging
    for df in (pg, adv, poss):
        df["Season"] = season

    # sanity: ORtg/DRtg should be in per_poss
    if "ORtg" not in poss.columns or "DRtg" not in poss.columns:
        raise ValueError(
            f"{team_slug} {season}: per_poss missing ORtg/DRtg. "
            f"Per_poss columns: {poss.columns.tolist()[:70]}"
        )

    # merge stepwise
    merged = pd.merge(pg, adv, on=["Player", "Season"], how="outer", suffixes=("_pg", "_adv"))
    merged = pd.merge(merged, poss[["Player", "Season", "ORtg", "DRtg"]], on=["Player", "Season"], how="left")

    return merged


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []

    for team_name, slug in TEAM_SLUGS.items():
        for season in SEASONS:
            print(f"Pulling stats: {team_name} ({slug}) {season}...")
            df = pull_team_stats(slug, season)
            df["Team"] = team_name
            frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_FILE, index=False)
    print(f"Saved: {OUT_FILE}")

    # quick check
    print("Has ORtg:", "ORtg" in out.columns, "| Has DRtg:", "DRtg" in out.columns)
    if "ORtg" in out.columns:
        nz = int((pd.to_numeric(out["ORtg"], errors="coerce").fillna(0) != 0).sum())
        print("Nonzero ORtg rows:", nz)


if __name__ == "__main__":
    main()