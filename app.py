import itertools
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# -----------------------------
# FILES (path-safe for Streamlit Cloud)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

PROFILES_CSV = BASE_DIR / "data" / "syracuse_player_profiles.csv"
ACC_PROFILES_CSV = BASE_DIR / "data" / "acc_player_profiles.csv"
MODEL_PATH = BASE_DIR / "data" / "lineup_model.joblib"

# Load ACC profiles
acc_profiles = pd.read_csv(ACC_PROFILES_CSV)

# Clean team names once
acc_profiles["Team_clean"] = (
    acc_profiles["Team"]
    .astype(str)
    .str.replace("*", "", regex=False)
    .str.strip()
)

# -----------------------------
# REQUIRED COLS IN PROFILES
# -----------------------------
REQUIRED_COLS = [
    "Player", "Season", "Pos", "MP_adv",
    "ORtg", "DRtg", "BPM",
    "TS%", "USG%", "AST%", "TOV%",
    "ORB%", "DRB%", "STL%", "BLK%",
]


# -----------------------------
# Data / Model loading
# -----------------------------
def ensure_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nHave: {list(df.columns)}")


def load_profiles() -> pd.DataFrame:
    df = pd.read_csv(PROFILES_CSV)
    ensure_columns(df, REQUIRED_COLS)
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
    df["MP_adv"] = pd.to_numeric(df["MP_adv"], errors="coerce")
    return df

def load_acc_profiles() -> pd.DataFrame:
    df = pd.read_csv(ACC_PROFILES_CSV)
    # allow ACC file to have more/different cols, but must have these
    need = ["Team", "Season", "Player", "Pos", "MP_adv"]
    ensure_columns(df, need)

    df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
    df["MP_adv"] = pd.to_numeric(df["MP_adv"], errors="coerce")

    # defensive: remove any Totals rows (Team Totals / School Totals etc.)
    df = df[~df["Player"].astype(str).str.contains("Totals", case=False, na=False)].copy()
    return df

def build_opponent_archetype_lineups(opp_pool_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Returns 4 opponent archetype lineups (each 5 players) from opponent pool df.
    Requires columns: Player, MP_adv, Pos, ORtg, DRtg, TS%, BPM, USG%
    """
    df = opp_pool_df.copy()

    # Ensure required cols exist
    for col in ["ORtg", "DRtg", "TS%", "BPM", "USG%"]:
        if col not in df.columns:
            df[col] = 0.0

    df["MP_adv"] = pd.to_numeric(df["MP_adv"], errors="coerce").fillna(0.0)
    df["ORtg"] = pd.to_numeric(df["ORtg"], errors="coerce").fillna(0.0)
    df["DRtg"] = pd.to_numeric(df["DRtg"], errors="coerce").fillna(0.0)
    df["TS%"] = pd.to_numeric(df["TS%"], errors="coerce").fillna(0.0)
    df["BPM"] = pd.to_numeric(df["BPM"], errors="coerce").fillna(0.0)
    df["USG%"] = pd.to_numeric(df["USG%"], errors="coerce").fillna(0.0)

    def pos_group(p) -> str:
        p = "" if pd.isna(p) else str(p).upper()
        if "C" in p:
            return "C"
        if "G" in p:
            return "G"
        return "F"

    df["PosGroup"] = df["Pos"].apply(pos_group)

    # Minute-weighted scores (keeps rotations realistic)
    # Small epsilon so low-min guys don’t spike
    w = (df["MP_adv"].clip(lower=1.0)) ** 0.75

    # “Big”: prefer C/F, rebounding/defense proxies via DRtg lower + BPM, keep some offense
    df["score_big"] = (
        (df["BPM"] * 1.0) +
        ((110.0 - df["DRtg"]) * 0.25) +
        (df["USG%"] * 0.02)
    ) * w + (df["PosGroup"].isin(["C", "F"]).astype(float) * 5.0)

    # “Small”: prefer G/F, spacing/creation
    df["score_small"] = (
        (df["BPM"] * 0.9) +
        (df["ORtg"] * 0.06) +
        (df["TS%"] * 1.5) +
        (df["USG%"] * 0.05)
    ) * w + (df["PosGroup"].isin(["G", "F"]).astype(float) * 5.0) - (df["PosGroup"].eq("C").astype(float) * 3.0)

    # “Shooting”: TS% and ORtg, lightly penalize low minutes
    df["score_shoot"] = (
        (df["TS%"] * 2.2) +
        (df["ORtg"] * 0.07) +
        (df["BPM"] * 0.5)
    ) * w

    # “Defense”: DRtg low is good + BPM + some usage isn’t required
    df["score_def"] = (
        ((110.0 - df["DRtg"]) * 0.35) +
        (df["BPM"] * 0.8) +
        (df["STL%"] * 0.3 if "STL%" in df.columns else 0.0) +
        (df["BLK%"] * 0.3 if "BLK%" in df.columns else 0.0)
    ) * w

    def pick5(score_col: str) -> list[str]:
        top = df.sort_values(score_col, ascending=False)["Player"].dropna().astype(str).tolist()
        # Deduplicate while preserving order
        seen = set()
        lineup = []
        for p in top:
            if p not in seen:
                lineup.append(p)
                seen.add(p)
            if len(lineup) == 5:
                break
        # fallback if somehow short
        if len(lineup) < 5:
            lineup = (df.sort_values("MP_adv", ascending=False)["Player"].dropna().astype(str).tolist())[:5]
        return lineup

    return {
        "Big": pick5("score_big"),
        "Small": pick5("score_small"),
        "Shooting": pick5("score_shoot"),
        "Defense": pick5("score_def"),
    }

# ---- LOAD DATA ONCE (global) ----
profiles = load_profiles()
acc_profiles = load_acc_profiles()

@st.cache_data
def cached_profiles() -> pd.DataFrame:
    return load_profiles()


@st.cache_resource
def cached_model():
    return joblib.load(MODEL_PATH)


# -----------------------------
# Feature engineering
# -----------------------------
def wavg(lineup_df: pd.DataFrame, w: pd.Series, col: str) -> float:
    x = pd.to_numeric(lineup_df[col], errors="coerce").astype(float)
    return float((x * w).sum() / w.sum())

def familiarity_score(ldf: pd.DataFrame) -> float:
    """
    Proxy for lineup 'comfort' based on minutes distribution.
    If one guy barely plays, lineup is less reliable.
    Returns multiplier in [0.75, 1.00].
    """
    mins = pd.to_numeric(ldf.get("MP_adv", 0), errors="coerce").fillna(0.0).astype(float)
    if len(mins) == 0:
        return 0.85

    avg_m = float(mins.mean())
    min_m = float(mins.min())

    if avg_m <= 0:
        return 0.85

    ratio = min_m / avg_m  # 0..1+
    mult = 0.75 + 0.25 * max(0.0, min(1.0, ratio))  # clamp into [0.75, 1.00]
    return float(mult)

def lineup_features(lineup_df: pd.DataFrame) -> dict:
    df = lineup_df.copy()

    # Ensure required numeric columns exist
    needed = [
        "MP_adv",
        "ORtg", "DRtg", "BPM",
        "TS%", "USG%",
        "AST%", "TOV%",
        "ORB%", "DRB%",
        "STL%", "BLK%",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = 0.0

    # numeric coercion
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    w = df["MP_adv"].fillna(0).astype(float)
    if w.sum() == 0:
        w = pd.Series([1.0] * len(df), index=df.index)

    def wavg(col: str) -> float:
        return float((df[col] * w).sum() / w.sum())

    ortg = wavg("ORtg")
    drtg = wavg("DRtg")

    feats = {
        "ORtg_lineup": ortg,
        "DRtg_lineup": drtg,
        "Net_lineup": ortg - drtg,  # convenience

        "BPM_lineup": wavg("BPM"),
        "TS_lineup": wavg("TS%"),
        "USG_lineup": wavg("USG%"),
        "USG_std_lineup": float(df["USG%"].std()) if len(df) > 1 else 0.0,
        "AST_lineup": wavg("AST%"),
        "TOV_lineup": wavg("TOV%"),
        "ORB_lineup": wavg("ORB%"),
        "DRB_lineup": wavg("DRB%"),
        "STL_lineup": wavg("STL%"),
        "BLK_lineup": wavg("BLK%"),

        "MIN_total": float(df["MP_adv"].sum()),
        "MIN_avg": float(df["MP_adv"].mean()),
    }
    return feats

def matchup_adjustment(syr_feats: dict, opp_feats: dict) -> float:
    """
    Level-1 matchup heuristic (returns a single float adjustment in net-rating points).
    """
    syr_DRtg = float(syr_feats.get("DRtg_lineup", 0.0))
    syr_TOV  = float(syr_feats.get("TOV_lineup", 0.0))
    syr_ORB  = float(syr_feats.get("ORB_lineup", 0.0))
    syr_DRB  = float(syr_feats.get("DRB_lineup", 0.0))

    opp_ORtg = float(opp_feats.get("ORtg_lineup", 0.0))
    opp_STL  = float(opp_feats.get("STL_lineup", 0.0))
    opp_ORB  = float(opp_feats.get("ORB_lineup", 0.0))
    opp_DRB  = float(opp_feats.get("DRB_lineup", 0.0))

    def_edge = (opp_ORtg - syr_DRtg) * 0.30
    tov_risk = (syr_TOV * (1.0 + 0.03 * opp_STL)) * 0.25
    reb_edge = ((syr_DRB + syr_ORB) - (opp_DRB + opp_ORB)) * 0.08

    adj = def_edge - tov_risk + reb_edge
    adj = max(-6.0, min(6.0, float(adj)))
    return adj

def matchup_breakdown(syr_feats: dict, opp_feats: dict) -> dict:
    adj = matchup_adjustment(syr_feats, opp_feats)

    syr_DRtg = float(syr_feats.get("DRtg_lineup", 0.0))
    syr_TOV  = float(syr_feats.get("TOV_lineup", 0.0))
    syr_ORB  = float(syr_feats.get("ORB_lineup", 0.0))
    syr_DRB  = float(syr_feats.get("DRB_lineup", 0.0))

    opp_ORtg = float(opp_feats.get("ORtg_lineup", 0.0))
    opp_STL  = float(opp_feats.get("STL_lineup", 0.0))
    opp_ORB  = float(opp_feats.get("ORB_lineup", 0.0))
    opp_DRB  = float(opp_feats.get("DRB_lineup", 0.0))

    def_edge = (opp_ORtg - syr_DRtg) * 0.30
    tov_risk = (syr_TOV * (1.0 + 0.03 * opp_STL)) * 0.25
    reb_edge = ((syr_DRB + syr_ORB) - (opp_DRB + opp_ORB)) * 0.08

    return {
        "matchup_adj": float(adj),
        "def_edge": float(def_edge),
        "tov_risk": float(tov_risk),
        "reb_edge": float(reb_edge),
    }

def score_lineup(
    season_df: pd.DataFrame,
    model,
    players: list[str],
    *,
    # constraints
    min_guards: int,
    min_bigs: int,
    max_centers: int,
    # reliability
    soft_min: float,
    scale: float,
    # fatigue
    minutes_left: int,
    fatigue_level: float,
    fatigue_strength: float,
    # opponent (optional)
    opp_df_all: pd.DataFrame | None = None,
    opp_lineup: list[str] | None = None,
    matchup_weight: float = 0.35,
) -> dict:

    players = [str(p) for p in players]

    ldf = season_df[season_df["Player"].astype(str).isin(players)].copy()
    if len(ldf) != 5:
        return {"valid": False, "reason": "missing_players"}

    feats = lineup_features(ldf)
    fam_mult = familiarity_score(ldf)

    # Opponent features
    opp_feats = None
    if opp_df_all is not None and opp_lineup is not None and len(opp_lineup) == 5:
        opp_ldf = opp_df_all[opp_df_all["Player"].astype(str).isin([str(p) for p in opp_lineup])].copy()
        if len(opp_ldf) == 5:
            opp_feats = lineup_features(opp_ldf)

    X = build_X_for_model(model, feats)
    pred_raw = float(model.predict(X)[0])

    pen, min_mp, avg_mp = reliability_penalty(ldf, soft_min=soft_min, scale=scale)

    # Fatigue multiplier
    late_factor = 0.5 + 0.5 * (1.0 - min(float(minutes_left) / 40.0, 1.0))  # 0.5 early -> 1.0 late
    usage_factor = min(float(avg_mp) / float(soft_min), 2.0)
    fatigue_mult = 1.0 - (fatigue_strength * fatigue_level * late_factor * usage_factor)
    fatigue_mult = max(0.70, float(fatigue_mult))

    pred_adj = (pred_raw - pen) * fatigue_mult

    # Chemistry shrink
    pred_adj = pred_adj * (0.85 + 0.15 * float(feats.get("Chemistry", 1.0)))
    pred_adj = float(pred_adj) * fam_mult

    # Matchup adjustment
    if opp_feats is not None:
        m = matchup_adjustment(feats, opp_feats)
        if isinstance(m, dict):
            pred_adj = float(pred_adj) + float(matchup_weight) * float(m.get("matchup_adj", 0.0))
            matchup_adj = float(m.get("matchup_adj", 0.0))
        else:
            pred_adj = float(pred_adj) + float(matchup_weight) * float(m)
            matchup_adj = float(m)
    else:
        matchup_adj = 0.0

    clutch_mult = 1.0
    if int(minutes_left) <= 5:
        # Reward efficiency + ball security late
        clutch_mult += 0.10 * float(feats.get("TS_lineup", 0.0))
        clutch_mult -= 0.05 * float(feats.get("TOV_lineup", 0.0))

    clutch_mult = max(0.85, min(1.10, clutch_mult))
    pred_adj = float(pred_adj) * clutch_mult

    counts = pos_counts(ldf)

    spacing_bonus = (
            0.6 * float(feats.get("TS_lineup", 0.0)) +
            0.15 * float(feats.get("BPM_lineup", 0.0)) +
            0.10 * float(counts.get("G", 0))
    )
    # small nudge only
    pred_adj = float(pred_adj) + float(spacing_bonus)

    if counts.get("G", 0) < min_guards:
        return {"valid": False, "reason": "min_guards"}
    if counts.get("BIG", 0) < min_bigs:
        return {"valid": False, "reason": "min_bigs"}
    if counts.get("C", 0) > max_centers:
        return {"valid": False, "reason": "max_centers"}

    trust = trust_multiplier(min_mp=min_mp, avg_mp=avg_mp, soft_min=float(soft_min))

    off_score = trust * fatigue_mult * (feats["ORtg_lineup"] + 10.0 * feats["TS_lineup"])
    def_score = trust * fatigue_mult * (
        -1.8 * feats["DRtg_lineup"]
        + 0.5 * feats["STL_lineup"]
        + 0.1 * feats["BLK_lineup"]
        + 0.3 * feats["DRB_lineup"]
    )
    sec_score = trust * fatigue_mult * (-feats["TOV_lineup"])
    reb_score = trust * fatigue_mult * (feats["ORB_lineup"] + feats["DRB_lineup"])

    return {
        "valid": True,
        "Players": " | ".join(players),
        "Predicted_Net": float(pred_adj),
        "Predicted_Net_raw": float(pred_raw),
        "Matchup_Adjustment": float(matchup_adj),
        "Penalty": float(pen),
        "Trust": float(trust),
        "Min_MP_in_lineup": float(min_mp),
        "Avg_MP_in_lineup": float(avg_mp),
        "ORtg_lineup": float(feats["ORtg_lineup"]),
        "DRtg_lineup": float(feats["DRtg_lineup"]),
        "Chemistry": float(feats.get("Chemistry", 1.0)),
        "Familiarity_mult": fam_mult,
        "Clutch_mult": clutch_mult,
        "Spacing_bonus": spacing_bonus,
        "MIN_total": float(feats["MIN_total"]),
        "G": int(counts.get("G", 0)),
        "F": int(counts.get("F", 0)),
        "C": int(counts.get("C", 0)),
        "_off": float(off_score),
        "_def": float(def_score),
        "_sec": float(sec_score),
        "_reb": float(reb_score),
        "Fatigue_mult": float(fatigue_mult),
    }

def _safe_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(float)

def _pos_group(pos: str) -> str:
    p = "" if pd.isna(pos) else str(pos).upper()
    if "C" in p:
        return "C"
    if "G" in p:
        return "G"
    return "F"

def _pos_counts(df: pd.DataFrame) -> dict:
    pos = df["Pos"].apply(_pos_group)
    return {
        "G": int((pos == "G").sum()),
        "F": int((pos == "F").sum()),
        "C": int((pos == "C").sum()),
        "BIG": int(((pos == "F") | (pos == "C")).sum()),
    }

def _pick_lineup_from_ranked(df_ranked: pd.DataFrame,
                            min_guards: int = 2,
                            min_bigs: int = 2,
                            max_centers: int = 2) -> list[str]:
    """
    Greedy pick from a ranked player list until we have 5 that meet a basic shape.
    """
    picked = []
    for _, row in df_ranked.iterrows():
        p = str(row["Player"])
        if p in picked:
            continue
        picked.append(p)
        if len(picked) == 5:
            tmp = df_ranked[df_ranked["Player"].isin(picked)].copy()
            counts = _pos_counts(tmp)
            if counts["G"] >= min_guards and counts["BIG"] >= min_bigs and counts["C"] <= max_centers:
                return picked
            # if shape fails, keep searching by swapping the last pick out
            picked.pop()

    # fallback: just top 5
    return df_ranked["Player"].head(5).astype(str).tolist()

def build_opponent_archetype_lineups(opp_pool_df: pd.DataFrame) -> dict:
    """
    Returns dict[str, list[str]] of archetype -> 5 players.

    Requires columns: Player, Pos, MP_adv, ORtg, DRtg, TS%, STL%, BLK%, DRB%, ORB%, TOV%
    (missing columns are treated as 0).
    """
    df = opp_pool_df.copy()

    # numeric columns (safe)
    df["MP_adv"] = _safe_num(df.get("MP_adv", 0))
    df["ORtg"] = _safe_num(df.get("ORtg", 0))
    df["DRtg"] = _safe_num(df.get("DRtg", 0))
    df["TS%"] = _safe_num(df.get("TS%", 0))
    df["STL%"] = _safe_num(df.get("STL%", 0))
    df["BLK%"] = _safe_num(df.get("BLK%", 0))
    df["DRB%"] = _safe_num(df.get("DRB%", 0))
    df["ORB%"] = _safe_num(df.get("ORB%", 0))
    df["TOV%"] = _safe_num(df.get("TOV%", 0))

    # minutes-weighted “reliability” bump so low-minute guys don’t dominate archetypes
    mp = df["MP_adv"].clip(lower=0)
    rel = (mp / (mp.max() + 1e-6)).clip(0, 1)  # 0..1
    df["_rel"] = rel

    # Scores (all higher = better for archetype)
    # Big: rim protection + boards + size proxy (C/F) + minutes
    df["_is_big"] = df["Pos"].apply(_pos_group).isin(["F", "C"]).astype(int)

    df["_big_score"] = (
        0.55 * df["_is_big"]
        + 0.25 * (df["DRB%"] + df["ORB%"])
        + 0.20 * (df["BLK%"])
        + 0.30 * df["_rel"]
    )

    # Small: guards + ball security + overall efficiency + minutes
    df["_is_guard"] = df["Pos"].apply(_pos_group).eq("G").astype(int)

    df["_small_score"] = (
        0.60 * df["_is_guard"]
        + 0.25 * (df["ORtg"] / 120.0)
        + 0.15 * (df["TS%"])
        - 0.20 * (df["TOV%"] / 30.0)
        + 0.30 * df["_rel"]
    )

    # Shooting: ORtg + TS% + low TOV + minutes (we don’t have 3PA rate consistently)
    df["_shoot_score"] = (
        0.45 * (df["ORtg"] / 120.0)
        + 0.45 * (df["TS%"])
        - 0.20 * (df["TOV%"] / 30.0)
        + 0.30 * df["_rel"]
    )

    # Defense: low DRtg + steals + blocks + defensive rebounding + minutes
    df["_def_score"] = (
        0.55 * ((120.0 - df["DRtg"]) / 40.0)   # lower DRtg => higher score
        + 0.20 * (df["STL%"] / 10.0)
        + 0.15 * (df["BLK%"] / 10.0)
        + 0.10 * (df["DRB%"] / 30.0)
        + 0.30 * df["_rel"]
    )

    # Rank and pick with archetype-specific shape constraints
    big_rank = df.sort_values(["_big_score", "MP_adv"], ascending=False)
    small_rank = df.sort_values(["_small_score", "MP_adv"], ascending=False)
    shoot_rank = df.sort_values(["_shoot_score", "MP_adv"], ascending=False)
    def_rank = df.sort_values(["_def_score", "MP_adv"], ascending=False)

    lineups = {
        "Big": _pick_lineup_from_ranked(big_rank, min_guards=1, min_bigs=3, max_centers=2),
        "Small": _pick_lineup_from_ranked(small_rank, min_guards=3, min_bigs=1, max_centers=1),
        "Shooting": _pick_lineup_from_ranked(shoot_rank, min_guards=2, min_bigs=2, max_centers=2),
        "Defense": _pick_lineup_from_ranked(def_rank, min_guards=2, min_bigs=2, max_centers=2),
    }
    return lineups

def objective_weights(priority: str, score_margin: int, minutes_left: int) -> dict:
    presets = {
        "Balanced":         {"net": 0.55, "off": 0.15, "def": 0.15, "sec": 0.10, "reb": 0.05},
        "Need a bucket":    {"net": 0.40, "off": 0.35, "def": 0.10, "sec": 0.10, "reb": 0.05},
        "Get stops":        {"net": 0.35, "off": 0.05, "def": 0.45, "sec": 0.10, "reb": 0.05},
        "Protect the ball": {"net": 0.40, "off": 0.10, "def": 0.10, "sec": 0.35, "reb": 0.05},
        "Win the glass":    {"net": 0.40, "off": 0.10, "def": 0.10, "sec": 0.05, "reb": 0.35},
    }
    w = presets[priority].copy()

    late = minutes_left <= 5
    close = abs(score_margin) <= 6
    trailing = score_margin < 0
    leading = score_margin > 0

    if late and close:
        w["sec"] += 0.05
        w["def"] += 0.05
        w["off"] -= 0.05
        w["net"] -= 0.05

    if late and trailing:
        w["off"] += 0.07
        w["sec"] += 0.03
        w["def"] -= 0.05
        w["reb"] -= 0.05

    if late and leading:
        w["def"] += 0.07
        w["sec"] += 0.05
        w["off"] -= 0.07
        w["reb"] -= 0.05

    s = sum(w.values())
    for k in w:
        w[k] = w[k] / s
    return w

def pos_counts(lineup_df: pd.DataFrame) -> dict:
    pos = lineup_df["Pos"].fillna("").astype(str).str.strip().str.upper()
    g = int((pos == "G").sum())
    f = int((pos == "F").sum())
    c = int((pos == "C").sum())
    big = int(((pos == "F") | (pos == "C")).sum())
    return {"G": g, "F": f, "C": c, "BIG": big}


def build_X_for_model(model, feats: dict) -> pd.DataFrame:
    X = pd.DataFrame([feats])
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        for c in cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[cols]
    return X


# -----------------------------
# Reliability penalty (same idea as rank_lineups)
# -----------------------------
def reliability_penalty(lineup_df: pd.DataFrame, soft_min: int, scale: float) -> tuple[float, float, float]:
    mps = lineup_df["MP_adv"].astype(float).tolist()
    min_mp = float(min(mps))
    avg_mp = float(sum(mps) / len(mps))

    if soft_min <= 0 or scale <= 0:
        return 0.0, min_mp, avg_mp

    # Part 1: weakest-link penalty
    gap_min = max(0.0, (soft_min - min_mp) / soft_min)
    p1 = scale * gap_min

    # Part 2: depth penalty (multiple sub-threshold guys)
    gaps = [max(0.0, (soft_min - mp) / soft_min) for mp in mps]
    p2 = (scale * 0.6) * (sum(gaps) / len(gaps))

    return float(p1 + p2), min_mp, avg_mp


# -----------------------------
# NEW: Trust multiplier (Fix #1)
# Shrinks component scores for less-proven lineups
# -----------------------------
def trust_multiplier(min_mp: float, avg_mp: float, soft_min: float) -> float:
    """
    Output range: 0.6 to 1.0.
    Uses both min and avg minutes so one low-minute guy drags trust down.
    """
    if soft_min <= 0:
        return 1.0

    a = min(1.0, max(0.0, min_mp / soft_min))
    b = min(1.0, max(0.0, avg_mp / soft_min))
    t = 0.65 * a + 0.35 * b

    return 0.6 + 0.4 * t


# -----------------------------
# Utility: min-max normalize within candidates
# -----------------------------
def minmax(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    lo, hi = float(series.min()), float(series.max())
    if hi - lo < 1e-9:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - lo) / (hi - lo)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Syracuse Lineup Evaluator", layout="wide")
st.title("Syracuse Lineup Evaluator")

tab_eval, tab_reco, tab_matchups, tab_rotation = st.tabs(
    ["Lineup Rankings", "Lineup Recommendations", "Matchups vs Opponent", "Rotations Inisghts"]
)
df = cached_profiles()
model = cached_model()

# Sidebar controls (shared by both tabs)
st.sidebar.header("Controls")

seasons = sorted([int(x) for x in df["Season"].dropna().unique().tolist()])
season = st.sidebar.selectbox("Season", seasons, index=len(seasons) - 1)

st.sidebar.subheader("Rotation pool")
min_pool_minutes = st.sidebar.slider("Min minutes to be eligible", 0, 1200, 250, step=25)
top_n_pool = st.sidebar.slider("Top N by minutes", 5, 20, 8, step=1)

st.sidebar.subheader("Reliability penalty")
soft_min = st.sidebar.slider("Soft minutes threshold", 0, 1200, 500, step=25)
scale = st.sidebar.slider("Penalty scale", 0.0, 10.0, 3.0, step=0.25)

st.sidebar.subheader("Positional constraints")
min_guards = st.sidebar.slider("Min guards", 0, 3, 2, step=1)
min_bigs = st.sidebar.slider("Min bigs (F or C)", 0, 3, 2, step=1)
max_centers = st.sidebar.slider("Max centers", 0, 3, 1, step=1)

# Sidebar filters already defined above:
# season, min_pool_minutes, top_n_pool

# STEP 1: build season_df
season_df = profiles[profiles["Season"] == season].copy()

season_df["MP_adv"] = pd.to_numeric(
    season_df.get("MP_adv", 0),
    errors="coerce"
).fillna(0.0)

# STEP 2: build pool_df
pool_df = (
    season_df[season_df["MP_adv"] >= min_pool_minutes]
    .sort_values("MP_adv", ascending=False)
    .head(top_n_pool)
    .copy()
)

# STEP 3: defensive fix
pool_df = pool_df[
    ~pool_df["Player"].astype(str).str.contains("Totals", case=False, na=False)
].copy()

# STEP 4: extract players
pool_players = pool_df["Player"].dropna().tolist()

st.caption(f"Model: `{MODEL_PATH}` | Profiles: `{PROFILES_CSV}`")


# -----------------------------
# TAB 1: Evaluator
# -----------------------------
with tab_eval:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Rotation pool (by minutes)")
        st.dataframe(pool_df[["Player", "Pos", "MP_adv"]], width="stretch")

    with c2:
        st.subheader("Evaluate a custom 5-man lineup")

        if len(pool_players) < 5:
            st.warning("Not enough players in the pool. Lower the minutes filter or raise Top N.")
        else:
            picks = st.multiselect("Select exactly 5 players", pool_players, default=pool_players[:5], max_selections=5)

            if len(picks) != 5:
                st.info("Pick 5 players to evaluate.")
            else:
                ldf = season_df[season_df["Player"].isin(picks)].copy()

                feats = lineup_features(ldf)
                X = build_X_for_model(model, feats)

                pred_raw = float(model.predict(X)[0])
                pen, min_mp, avg_mp = reliability_penalty(ldf, soft_min=soft_min, scale=scale)
                pred_adj = pred_raw - pen


                counts = pos_counts(ldf)

                passes = True
                if counts["G"] < min_guards:
                    passes = False
                if counts["BIG"] < min_bigs:
                    passes = False
                if counts["C"] > max_centers:
                    passes = False

                st.metric("Predicted Net (Adjusted)", f"{pred_adj:.2f}")

                st.write({
                    "Predicted_Net_raw": round(pred_raw, 2),
                    "Penalty": round(pen, 3),
                    "Min_MP_in_lineup": round(min_mp, 1),
                    "Avg_MP_in_lineup": round(avg_mp, 1),
                    "ORtg_lineup": round(feats["ORtg_lineup"], 2),
                    "DRtg_lineup": round(feats["DRtg_lineup"], 2),
                    "MIN_total": round(feats["MIN_total"], 1),
                    "G": counts["G"],
                    "F": counts["F"],
                    "C": counts["C"],
                    "Passes_constraints": passes,
                })

    st.markdown("---")
    st.subheader("Top predicted 5-man lineups from the pool")

    must_include = st.selectbox("Must include (optional)", ["(none)"] + pool_players, index=0)

    combos = list(itertools.combinations(pool_players, 5))
    rows = []

    for combo in combos:
        if must_include != "(none)" and must_include not in combo:
            continue

        ldf = season_df[season_df["Player"].isin(combo)].copy()
        feats = lineup_features(ldf)
        X = build_X_for_model(model, feats)

        pred_raw = float(model.predict(X)[0])
        pen, min_mp, avg_mp = reliability_penalty(ldf, soft_min=soft_min, scale=scale)
        pred_adj = pred_raw - pen

        counts = pos_counts(ldf)

        if counts["G"] < min_guards:
            continue
        if counts["BIG"] < min_bigs:
            continue
        if counts["C"] > max_centers:
            continue

        rows.append({
            "Players": " | ".join(combo),
            "Predicted_Net": pred_adj,
            "Predicted_Net_raw": pred_raw,
            "Penalty": pen,
            "Min_MP_in_lineup": min_mp,
            "Avg_MP_in_lineup": avg_mp,
            "ORtg_lineup": feats["ORtg_lineup"],
            "DRtg_lineup": feats["DRtg_lineup"],
            "MIN_total": feats["MIN_total"],
            "G": counts["G"],
            "F": counts["F"],
            "C": counts["C"],
        })

    if not rows:
        st.warning("No lineups to show. Expand pool or loosen constraints.")
    else:
        out = pd.DataFrame(rows).sort_values("Predicted_Net", ascending=False)
        st.dataframe(out.head(20), width="stretch")


# -----------------------------
# TAB 2: Recommendations (Fix #1 applied)
# -----------------------------
with tab_reco:
    st.subheader("Lineup Recommendations (Which lineup when?)")

    # -----------------------------
    # Opponent dropdown (ACC, exclude Syracuse)
    # -----------------------------
    if "Team_clean" not in acc_profiles.columns:
        acc_profiles["Team_clean"] = (
            acc_profiles["Team"]
            .astype(str)
            .str.replace("*", "", regex=False)
            .str.strip()
        )

    acc_teams = sorted(
        acc_profiles["Team_clean"]
        .dropna()
        .unique()
        .tolist()
    )
    acc_teams = [t for t in acc_teams if str(t).strip().casefold() != "syracuse"]

    opp_team = st.selectbox(
        "Opponent (ACC)",
        acc_teams,
        index=0,
        key="reco_opp_team",
    )

    if len(pool_players) < 5:
        st.warning("Not enough players in the pool. Lower the minutes filter or raise Top N.")
        st.stop()

    # -----------------------------
    # Game-state controls
    # -----------------------------
    cA, cB, cC = st.columns(3)
    with cA:
        score_margin = st.slider(
            "Score margin (Syracuse - Opponent)", -25, 25, 0, step=1, key="reco_margin"
        )
    with cB:
        minutes_left = st.slider(
            "Minutes remaining", 0, 40, 8, step=1, key="reco_minutes"
        )
    with cC:
        priority = st.selectbox(
            "Primary priority",
            ["Balanced", "Need a bucket", "Get stops", "Protect the ball", "Win the glass"],
            index=0,
            key="reco_priority",
        )

    # -----------------------------
    # Fatigue controls
    # -----------------------------
    fA, fB = st.columns(2)
    with fA:
        fatigue_level = st.slider(
            "Fatigue level (0 = fresh, 1 = gassed)",
            0.0, 1.0, 0.25, 0.05,
            key="reco_fatigue_level",
        )
    with fB:
        fatigue_strength = st.slider(
            "Fatigue impact strength",
            0.00, 0.30, 0.12, 0.01,
            key="reco_fatigue_strength",
        )

    weights = objective_weights(priority, int(score_margin), int(minutes_left))
    st.write({"weights": {k: round(v, 3) for k, v in weights.items()}})

    # -----------------------------
    # Opponent season = SAME as current season (tab_reco only)
    # -----------------------------
    opp_df_all = acc_profiles[
        (acc_profiles["Team_clean"] == opp_team) &
        (acc_profiles["Season"] == int(season))
    ].copy()

    # remove Team Totals / Totals rows
    if "Player" in opp_df_all.columns:
        opp_df_all = opp_df_all[
            ~opp_df_all["Player"].astype(str).str.contains("Totals", case=False, na=False)
        ].copy()

    if opp_df_all.empty:
        st.warning(f"No opponent data for {opp_team} in {int(season)}.")
        st.stop()

    opp_df_all["MP_adv"] = pd.to_numeric(opp_df_all.get("MP_adv", 0), errors="coerce").fillna(0.0)

    opp_pool_df = (
        opp_df_all
        .sort_values("MP_adv", ascending=False)
        .head(int(top_n_pool))
        .copy()
    )
    opp_pool_players = opp_pool_df["Player"].dropna().astype(str).tolist()

    st.markdown("### Opponent lineup")

    if len(opp_pool_players) < 5:
        st.warning("Opponent pool has < 5 players. Raise Top N or confirm opponent data.")
        st.stop()

    opp_mode = st.radio(
        "Opponent lineup mode",
        ["Auto (top 5 by minutes)", "Archetype (Big / Small / Shooting / Defense)", "Manual pick 5"],
        horizontal=True,
        key="reco_opp_mode",
    )

    if opp_mode.startswith("Auto"):
        opp_lineup = opp_pool_players[:5]
        st.caption("Using opponent top-5 by minutes.")

    elif opp_mode.startswith("Archetype"):
        archetypes = build_opponent_archetype_lineups(opp_pool_df)
        archetype = st.selectbox(
            "Opponent archetype",
            ["Big", "Small", "Shooting", "Defense"],
            index=0,
            key="reco_opp_arch",
        )
        opp_lineup = archetypes[archetype]
        st.caption(f"Using opponent archetype: {archetype}")

    else:
        opp_lineup = st.multiselect(
            "Select 5 opponent players",
            options=opp_pool_players,
            default=opp_pool_players[:5],
            max_selections=5,
            key="reco_opp_manual",
        )
        if len(opp_lineup) != 5:
            st.warning("Pick exactly 5 opponent players.")
            st.stop()

    st.write({"Opponent_lineup": " | ".join([str(p) for p in opp_lineup])})

    # -----------------------------
    # Score all 5-man combos
    # -----------------------------
    combos = list(itertools.combinations(pool_players, 5))

    rows = []
    fail_reasons = {"missing_players": 0, "min_guards": 0, "min_bigs": 0, "max_centers": 0}

    for combo in combos:
        res = score_lineup(
            season_df=season_df,
            model=model,
            players=list(combo),
            min_guards=min_guards,
            min_bigs=min_bigs,
            max_centers=max_centers,
            soft_min=float(soft_min),
            scale=float(scale),
            minutes_left=int(minutes_left),
            fatigue_level=float(fatigue_level),
            fatigue_strength=float(fatigue_strength),
            opp_df_all=opp_df_all,
            opp_lineup=opp_lineup,
        )
        if not res.get("valid", False):
            r = res.get("reason")
            if r in fail_reasons:
                fail_reasons[r] += 1
            continue
        rows.append(res)

    if not rows:
        st.error("No candidate lineups passed constraints.")
        st.write({
            "Season": int(season),
            "Pool_size": int(len(pool_players)),
            "Combos_possible": int(len(combos)),
            "Fails": fail_reasons,
            "Try": "Lower constraints or expand pool (min minutes / Top N)."
        })
        st.stop()

    cand = pd.DataFrame(rows)

    # Normalize within candidates so the Objective score is comparable
    cand["net_n"] = minmax(cand["Predicted_Net"])
    cand["off_n"] = minmax(cand["_off"])
    cand["def_n"] = minmax(cand["_def"])
    cand["sec_n"] = minmax(cand["_sec"])
    cand["reb_n"] = minmax(cand["_reb"])

    cand["Objective"] = (
        weights["net"] * cand["net_n"]
        + weights["off"] * cand["off_n"]
        + weights["def"] * cand["def_n"]
        + weights["sec"] * cand["sec_n"]
        + weights["reb"] * cand["reb_n"]
    )

    ranked = cand.sort_values(["Objective", "Predicted_Net"], ascending=False).reset_index(drop=True)

    display_cols = [
        "Players",
        "Objective",
        "Predicted_Net", "Predicted_Net_raw", "Penalty", "Trust",
        "ORtg_lineup", "DRtg_lineup",
        "Min_MP_in_lineup", "Avg_MP_in_lineup",
        "G", "F", "C",
    ]

    st.markdown("### Recommended lineups")
    st.dataframe(ranked.head(20)[display_cols], width="stretch")

    st.download_button(
        "Download Recommended Lineups (CSV)",
        ranked.to_csv(index=False).encode("utf-8"),
        file_name=f"reco_{int(season)}_{str(opp_team).replace(' ', '_')}.csv",
        mime="text/csv",
        key="dl_reco_csv",
    )

    # -----------------------------
    # Why this top lineup?
    # -----------------------------
    st.markdown("### Why This Top Lineup?")

    top_lineup = ranked.iloc[0]

    drivers = {
        "Offense Impact": float(top_lineup.get("_off", 0.0)),
        "Defensive Impact": float(top_lineup.get("_def", 0.0)),
        "Ball Security": float(top_lineup.get("_sec", 0.0)),
        "Rebounding Edge": float(top_lineup.get("_reb", 0.0)),
        "Matchup Adjustment": float(top_lineup.get("Matchup_Adjust", 0.0)),
    }

    top3 = sorted(drivers.items(), key=lambda x: x[1], reverse=True)[:3]
    for name, _ in top3:
        st.write(f"• {name} drove this lineup’s projected edge.")

    # -----------------------------
    # Substitution Assistant
    # -----------------------------
    st.markdown("---")
    st.subheader("Substitution Assistant (1-sub options)")

    st.caption("Pick a current 5. This will suggest the best one-for-one swaps from the pool for the current priority/game state.")

    current = st.multiselect(
        "Current 5 on the floor",
        pool_players,
        default=pool_players[:5],
        max_selections=5,
        key="sub_current",
    )

    lock = st.multiselect(
        "Lock players (optional) — they will NOT be subbed out",
        current,
        default=[],
        key="sub_lock",
    )

    top_k = st.slider("Show top swaps", 5, 30, 15, step=1, key="sub_topk")

    if len(current) != 5:
        st.info("Select exactly 5 players to get substitution recommendations.")
        st.stop()

    base = score_lineup(
        season_df=season_df,
        model=model,
        players=current,
        min_guards=min_guards,
        min_bigs=min_bigs,
        max_centers=max_centers,
        soft_min=float(soft_min),
        scale=float(scale),
        minutes_left=int(minutes_left),
        fatigue_level=float(fatigue_level),
        fatigue_strength=float(fatigue_strength),
        opp_df_all=opp_df_all,
        opp_lineup=opp_lineup,
    )

    if not base.get("valid", False):
        st.error("Current lineup fails constraints. Adjust the 5 or loosen constraints.")
        st.stop()

    bench = [p for p in pool_players if p not in current]
    subs = []

    for out_p in current:
        if out_p in lock:
            continue
        for in_p in bench:
            new5 = [p for p in current if p != out_p] + [in_p]
            res = score_lineup(
                season_df=season_df,
                model=model,
                players=new5,
                min_guards=min_guards,
                min_bigs=min_bigs,
                max_centers=max_centers,
                soft_min=float(soft_min),
                scale=float(scale),
                minutes_left=int(minutes_left),
                fatigue_level=float(fatigue_level),
                fatigue_strength=float(fatigue_strength),
                opp_df_all=opp_df_all,
                opp_lineup=opp_lineup,
            )
            if not res.get("valid", False):
                continue

            subs.append({
                "OUT": out_p,
                "IN": in_p,
                "New_Lineup": res["Players"],
                "Predicted_Net": res["Predicted_Net"],
                "Delta_Predicted_Net": float(res["Predicted_Net"]) - float(base["Predicted_Net"]),
                "Penalty": res["Penalty"],
                "Trust": res["Trust"],
                "_off": res["_off"],
                "_def": res["_def"],
                "_sec": res["_sec"],
                "_reb": res["_reb"],
            })

    if not subs:
        st.warning("No valid 1-sub swaps found (constraints too tight or pool too small).")
        st.stop()

    subs_df = pd.DataFrame(subs)

    subs_df["net_n"] = minmax(subs_df["Predicted_Net"])
    subs_df["off_n"] = minmax(subs_df["_off"])
    subs_df["def_n"] = minmax(subs_df["_def"])
    subs_df["sec_n"] = minmax(subs_df["_sec"])
    subs_df["reb_n"] = minmax(subs_df["_reb"])

    subs_df["Objective"] = (
        weights["net"] * subs_df["net_n"]
        + weights["off"] * subs_df["off_n"]
        + weights["def"] * subs_df["def_n"]
        + weights["sec"] * subs_df["sec_n"]
        + weights["reb"] * subs_df["reb_n"]
    )

    show_subs = subs_df.sort_values(["Objective", "Delta_Predicted_Net"], ascending=False).head(int(top_k))
    show_subs = show_subs[[
        "OUT", "IN", "New_Lineup",
        "Objective", "Delta_Predicted_Net",
        "Predicted_Net", "Penalty", "Trust",
    ]]

    st.markdown("### Best 1-sub options")
    st.dataframe(show_subs, width="stretch")

    st.download_button(
        "Download Substitution Suggestions (CSV)",
        subs_df.sort_values(["Objective", "Delta_Predicted_Net"], ascending=False).to_csv(index=False).encode("utf-8"),
        file_name=f"subs_{int(season)}_{str(opp_team).replace(' ', '_')}.csv",
        mime="text/csv",
        key="dl_subs_csv",
    )

with tab_matchups:
    st.subheader("Matchups vs Opponent (ACC)")

    if "acc_profiles" not in globals():
        st.error("acc_profiles not found. Make sure you load data/acc_player_profiles.csv into acc_profiles before this tab.")
        st.stop()

    # -----------------------------
    # Clean team names once (local)
    # -----------------------------
    if "Team_clean" not in acc_profiles.columns:
        acc_profiles["Team_clean"] = (
            acc_profiles["Team"]
            .astype(str)
            .str.replace("*", "", regex=False)
            .str.strip()
        )

    # -----------------------------
    # UI: Opponent + pool settings
    # -----------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        opp_team_list = sorted(
            acc_profiles["Team_clean"]
            .dropna()
            .unique()
            .tolist()
        )
        # remove Syracuse (case-insensitive)
        opp_team_list = [t for t in opp_team_list if str(t).strip().casefold() != "syracuse"]

        opp_team = st.selectbox("Opponent team", opp_team_list, index=0, key="match_opp_team")

    with c2:
        opp_season = st.selectbox(
            "Opponent season",
            sorted(acc_profiles["Season"].dropna().unique().astype(int).tolist()),
            index=0,
            key="match_opp_season",
        )

    with c3:
        opp_top_n_pool = st.slider(
            "Opponent Top N by minutes",
            5, 20, int(top_n_pool), step=1,
            key="match_opp_topn",
        )

    opp_min_pool_minutes = st.slider(
        "Opponent min minutes",
        0, 800, 0, step=25,
        key="match_opp_minmin",
    )

    # -----------------------------
    # Opponent rows
    # -----------------------------
    opp_df_all = acc_profiles[
        (acc_profiles["Team_clean"] == opp_team) &
        (acc_profiles["Season"] == int(opp_season))
    ].copy()

    # Remove Totals / Team Totals rows (treat as not-a-player)
    if "Player" in opp_df_all.columns:
        opp_df_all = opp_df_all[
            ~opp_df_all["Player"].astype(str).str.contains("Totals", case=False, na=False)
        ].copy()

    if opp_df_all.empty:
        st.warning("No opponent rows found for that team/season.")
        st.stop()

    opp_df_all["MP_adv"] = pd.to_numeric(opp_df_all.get("MP_adv", 0), errors="coerce").fillna(0.0)

    opp_pool_df = (
        opp_df_all[opp_df_all["MP_adv"] >= float(opp_min_pool_minutes)]
        .sort_values("MP_adv", ascending=False)
        .head(int(opp_top_n_pool))
        .copy()
    )
    opp_pool_players = opp_pool_df["Player"].dropna().astype(str).tolist()

    st.caption(f"Opponent pool size: {len(opp_pool_players)}")

    # -----------------------------
    # Opponent lineup selection
    # -----------------------------
    st.markdown("### Opponent lineup")

    if len(opp_pool_players) < 5:
        st.warning("Not enough opponent players in pool. Lower opponent min minutes or raise Opponent Top N.")
        st.stop()

    opp_mode = st.radio(
        "Opponent lineup mode",
        ["Auto (top 5 by minutes)", "Archetype (Big / Small / Shooting / Defense)", "Manual pick 5"],
        horizontal=True,
        key="match_opp_mode",
    )

    if opp_mode.startswith("Auto"):
        opp_lineup = opp_pool_players[:5]
        st.caption("Using opponent top-5 by minutes.")

    elif opp_mode.startswith("Archetype"):
        archetypes = build_opponent_archetype_lineups(opp_pool_df)
        archetype = st.selectbox(
            "Opponent archetype",
            ["Big", "Small", "Shooting", "Defense"],
            index=0,
            key="match_opp_arch",
        )
        opp_lineup = archetypes[archetype]
        st.caption(f"Using opponent archetype: {archetype}")

    else:
        opp_lineup = st.multiselect(
            "Select 5 opponent players",
            options=opp_pool_players,
            default=opp_pool_players[:5],
            max_selections=5,
            key="match_opp_manual",
        )
        if len(opp_lineup) != 5:
            st.warning("Pick exactly 5 opponent players.")
            st.stop()

    st.write({"Opponent lineup": " | ".join([str(p) for p in opp_lineup])})

    # -----------------------------
    # Syracuse lineup search settings
    # -----------------------------
    st.markdown("### Syracuse counter lineups (search)")

    if "pool_players" not in globals() or len(pool_players) < 5:
        st.error("pool_players not found (or < 5). Make sure your Syracuse pool is defined before the tabs.")
        st.stop()

    cA, cB, cC, cD = st.columns(4)
    with cA:
        min_guards = st.slider("Min guards (Syr)", 0, 5, 2, step=1, key="match_min_guards")
    with cB:
        min_bigs = st.slider("Min bigs (Syr)", 0, 5, 2, step=1, key="match_min_bigs")
    with cC:
        max_centers = st.slider("Max centers (Syr)", 0, 5, 2, step=1, key="match_max_centers")
    with cD:
        top_k = st.slider("Show top K results", 5, 50, 15, step=1, key="match_top_k")

    # Must exist in app.py
    if "lineup_features" not in globals():
        st.error("lineup_features() not found. It must exist in app.py.")
        st.stop()

    # -----------------------------
    # Local scoring function
    # -----------------------------
    def score_matchup(
        syr_players: list[str],
        opp_players: list[str],
        matchup_weight: float = 1.0,
    ) -> dict:
        players5 = [str(p) for p in syr_players]
        opp5 = [str(p) for p in opp_players]

        syr_ldf = season_df[season_df["Player"].astype(str).isin(players5)].copy()
        if len(syr_ldf) != 5:
            return {"valid": False, "reason": "missing_syr_players"}

        opp_ldf = opp_df_all[opp_df_all["Player"].astype(str).isin(opp5)].copy()
        if len(opp_ldf) != 5:
            return {"valid": False, "reason": "missing_opp_players"}

        counts = pos_counts(syr_ldf)
        if counts.get("G", 0) < int(min_guards):
            return {"valid": False, "reason": "min_guards"}
        if counts.get("BIG", 0) < int(min_bigs):
            return {"valid": False, "reason": "min_bigs"}
        if counts.get("C", 0) > int(max_centers):
            return {"valid": False, "reason": "max_centers"}

        syr_feats = lineup_features(syr_ldf)
        opp_feats = lineup_features(opp_ldf)

        X = build_X_for_model(model, syr_feats)
        pred_raw = float(model.predict(X)[0])

        pen, min_mp, avg_mp = reliability_penalty(
            syr_ldf,
            soft_min=float(soft_min),
            scale=float(scale),
        )
        pred_adj = pred_raw - float(pen)

        # fatigue (uses global sliders if they exist; otherwise defaults)
        # IMPORTANT: minutes_left/fatigue_level/fatigue_strength must exist globally in app
        late_factor = 0.5 + 0.5 * (1.0 - min(float(minutes_left) / 40.0, 1.0))
        usage_factor = min(float(avg_mp) / float(soft_min), 2.0)
        fatigue_mult = 1.0 - (float(fatigue_strength) * float(fatigue_level) * late_factor * usage_factor)
        fatigue_mult = max(0.70, float(fatigue_mult))
        pred_adj = float(pred_adj) * fatigue_mult

        chemistry_mult = 0.85 + 0.15 * float(syr_feats.get("Chemistry", 1.0))
        pred_adj = float(pred_adj) * float(chemistry_mult)

        fam_mult = familiarity_score(syr_ldf)
        pred_adj = float(pred_adj) * float(fam_mult)

        clutch_mult = 1.0
        if int(minutes_left) <= 5:
            clutch_mult += 0.10 * float(syr_feats.get("TS_lineup", 0.0))
            clutch_mult -= 0.05 * float(syr_feats.get("TOV_lineup", 0.0))
        clutch_mult = max(0.85, min(1.10, float(clutch_mult)))
        pred_adj = float(pred_adj) * clutch_mult

        spacing_bonus = (
            0.6 * float(syr_feats.get("TS_lineup", 0.0))
            + 0.15 * float(syr_feats.get("BPM_lineup", 0.0))
            + 0.10 * float(counts.get("G", 0))
        )
        pred_adj = float(pred_adj) + float(spacing_bonus)

        match_adj = float(matchup_adjustment(syr_feats, opp_feats))
        matchup_score = float(pred_adj) + float(matchup_weight) * match_adj

        return {
            "valid": True,
            "Syracuse_Lineup": " | ".join(players5),
            "Opponent_Lineup": " | ".join(opp5),
            "Matchup_Score": float(matchup_score),
            "Syr_Pred_Net": float(pred_adj),
            "Matchup_Adjust": float(match_adj),
            "Predicted_Net_raw": float(pred_raw),
            "Penalty": float(pen),
            "Min_MP_in_lineup": float(min_mp),
            "Avg_MP_in_lineup": float(avg_mp),
            "ORtg_lineup": float(syr_feats.get("ORtg_lineup", 0.0)),
            "DRtg_lineup": float(syr_feats.get("DRtg_lineup", 0.0)),
            "G": int(counts.get("G", 0)),
            "F": int(counts.get("F", 0)),
            "C": int(counts.get("C", 0)),
            "Fatigue_mult": float(fatigue_mult),
            "Chemistry_mult": float(chemistry_mult),
            "Familiarity_mult": float(fam_mult),
            "Clutch_mult": float(clutch_mult),
            "Spacing_bonus": float(spacing_bonus),
        }

    # -----------------------------
    # Run search
    # -----------------------------
    import itertools

    combos = list(itertools.combinations([str(p) for p in pool_players], 5))
    st.caption(f"Syracuse combos possible: {len(combos)}")

    run = st.button("Run matchup search", type="primary", key="run_matchups")

    if run:
        results = []
        fail = {"missing_syr_players": 0, "missing_opp_players": 0, "min_guards": 0, "min_bigs": 0, "max_centers": 0, "other": 0}

        prog = st.progress(0)
        n = len(combos)

        for i, combo in enumerate(combos, start=1):
            res = score_matchup(list(combo), opp_lineup)
            if not res.get("valid", False):
                r = res.get("reason", "other")
                fail[r] = fail.get(r, 0) + 1
            else:
                results.append(res)

            if i % max(1, n // 100) == 0:
                prog.progress(min(1.0, i / n))

        prog.progress(1.0)

        if not results:
            st.error("No Syracuse lineups passed constraints for this opponent lineup.")
            st.write("Fail counts:", fail)
            st.stop()

        ranked = (
            pd.DataFrame(results)
            .sort_values("Matchup_Score", ascending=False)
            .reset_index(drop=True)
        )

        st.download_button(
            "Download Matchup Results (CSV)",
            ranked.to_csv(index=False).encode("utf-8"),
            file_name=f"matchups_{str(opp_team).replace(' ', '_')}_{int(opp_season)}.csv",
            mime="text/csv",
            key="dl_matchups_csv",
        )

        st.markdown("### Best Syracuse counters vs this opponent lineup")
        st.dataframe(ranked.head(int(top_k)), width="stretch")

        st.markdown("### Why the #1 counter lineup?")
        top = ranked.iloc[0]
        st.write({
            "Matchup_Score": round(float(top["Matchup_Score"]), 3),
            "Syr_Pred_Net (after penalties/fatigue/etc.)": round(float(top["Syr_Pred_Net"]), 3),
            "Matchup_Adjust": round(float(top["Matchup_Adjust"]), 3),
            "Penalty": round(float(top["Penalty"]), 3),
            "Fatigue_mult": round(float(top["Fatigue_mult"]), 3),
            "Chemistry_mult": round(float(top["Chemistry_mult"]), 3),
            "Familiarity_mult": round(float(top["Familiarity_mult"]), 3),
            "Clutch_mult": round(float(top["Clutch_mult"]), 3),
            "Spacing_bonus": round(float(top["Spacing_bonus"]), 3),
        })

        st.markdown("### Worst Syracuse lineups vs this opponent lineup")
        st.dataframe(
            ranked.tail(int(min(top_k, 15))).sort_values("Matchup_Score", ascending=True),
            width="stretch"
        )

        out_path = f"data/matchups_{str(opp_team).replace(' ', '_').replace('(', '').replace(')', '')}_{int(opp_season)}.csv"
        ranked.to_csv(out_path, index=False)
        st.success(f"Saved: {out_path}")

with tab_rotation:
    st.subheader("Rotation Insights")

    if "cand" not in locals() and "cand" not in globals():
        st.info("Generate lineup recommendations first to populate insights.")
        st.stop()

    # Use most recent candidate DataFrame
    df = cand.copy()

    st.markdown("### Best Overall Lineup")
    best_overall = df.sort_values("Predicted_Net", ascending=False).iloc[0]
    st.write(best_overall["Players"])
    st.write(f"Predicted Net: {best_overall['Predicted_Net']:.2f}")

    st.markdown("### Best Defensive Lineup")
    best_def = df.sort_values("_def", ascending=False).iloc[0]
    st.write(best_def["Players"])

    st.markdown("### Best Offensive Lineup")
    best_off = df.sort_values("_off", ascending=False).iloc[0]
    st.write(best_off["Players"])

    st.markdown("### Safest (Most Reliable) Lineup")
    safest = df.sort_values("Trust", ascending=False).iloc[0]
    st.write(safest["Players"])

    st.markdown("### Player Appearance in Top 20 Lineups")
    top20 = df.sort_values("Predicted_Net", ascending=False).head(20)

    player_counts = {}
    for lineup in top20["Players"]:
        for p in lineup.split("|"):
            p = p.strip()
            player_counts[p] = player_counts.get(p, 0) + 1

    appearance_df = pd.DataFrame(
        player_counts.items(),
        columns=["Player", "Appearances_in_Top20"]
    ).sort_values("Appearances_in_Top20", ascending=False)

    st.dataframe(df, width="stretch")
