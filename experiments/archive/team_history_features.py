# -*- coding: utf-8 -*-
"""
team_history_features.py
========================
Leak-free historical team features for match duration prediction.

Features computed per match (using ONLY matches that ended BEFORE start_time):
  - team_avg_duration_N       : avg duration of last N games (window=20)
  - team_pace_score_N         : % of games that went > 40 min (late-game tendency)
  - team_snowball_factor_N    : avg radiant_gold_adv at min-15 in WON games
  - team_tempo_std_N          : std of durations (consistency proxy)

All features are computed for both radiant and dire.
Interaction features added at the end.

Usage:
  from experiments.team_history_features import build_team_history_features
  df = build_team_history_features(db_path, window=20)
  # then merge with your existing processor output on match_id
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW       = 20       # last N games per team
LATE_THRESH  = 40.0     # minutes — "late game" cutoff for pace_score
MIN15_IDX    = 14       # index 14 = minute 15 in radiant_gold_adv array

DEFAULT_DURATION = 40.0      # global mean, used when no history
DEFAULT_PACE     = 0.5
DEFAULT_SNOWBALL = 0.0
DEFAULT_TEMPO_STD = 5.0


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_team_history_features(
    db_path: str | Path,
    window: int = WINDOW,
    late_thresh: float = LATE_THRESH,
) -> pd.DataFrame:
    """
    Reads all matches from SQLite, sorts by start_time, and computes
    leak-free rolling team features for every match.

    Returns a DataFrame indexed by match_id with columns:
      rad_hist_avg_duration, rad_hist_pace_score, rad_hist_snowball_factor,
      rad_hist_tempo_std, rad_hist_games_played,
      dire_hist_avg_duration, dire_hist_pace_score, dire_hist_snowball_factor,
      dire_hist_tempo_std, dire_hist_games_played,
      (+ interaction features)
    """
    db_path = Path(db_path)
    assert db_path.exists(), f"DB not found: {db_path}"

    # -- Load raw matches --
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT raw_data FROM match_details").fetchall()
    conn.close()

    records = []
    for (raw,) in rows:
        try:
            m = json.loads(raw)
            mid      = m.get("match_id")
            st       = m.get("start_time")
            dur      = m.get("duration", 0)
            rwin     = m.get("radiant_win", False)
            rid      = m.get("radiant_team_id")
            did      = m.get("dire_team_id")
            gold_adv = m.get("radiant_gold_adv") or []

            if not (mid and st and dur > 0):
                continue

            # Gold advantage at minute 15 (from radiant perspective)
            gold_at_15 = gold_adv[MIN15_IDX] if len(gold_adv) > MIN15_IDX else None

            records.append({
                "match_id":      mid,
                "start_time":    st,
                "duration_min":  round(dur / 60, 2),
                "radiant_win":   bool(rwin),
                "radiant_tid":   rid,
                "dire_tid":      did,
                "gold_at_15":    gold_at_15,  # positive = radiant ahead
            })
        except Exception:
            continue

    df_raw = pd.DataFrame(records).sort_values("start_time").reset_index(drop=True)
    print(f"Loaded {len(df_raw)} matches for history computation.")

    # ------------------------------------------------------------------
    # Rolling history: per-team accumulator
    # State is updated AFTER computing features for each match (no leakage)
    # ------------------------------------------------------------------

    # team_id -> deque of dicts with {duration_min, won, gold_at_15}
    team_hist: defaultdict[int, list[dict]] = defaultdict(list)

    result_rows = []

    for _, row in df_raw.iterrows():
        rid = row["radiant_tid"]
        did = row["dire_tid"]

        rad_feats = _compute_team_feats(team_hist.get(rid, []), window, late_thresh, prefix="rad")
        dir_feats = _compute_team_feats(team_hist.get(did, []), window, late_thresh, prefix="dire")

        out = {"match_id": row["match_id"]}
        out.update(rad_feats)
        out.update(dir_feats)

        # --- Interaction features ---
        out["pace_diff"]          = rad_feats["rad_hist_pace_score"]  - dir_feats["dire_hist_pace_score"]
        out["avg_duration_diff"]  = rad_feats["rad_hist_avg_duration"] - dir_feats["dire_hist_avg_duration"]
        out["snowball_diff"]      = rad_feats["rad_hist_snowball_factor"] - dir_feats["dire_hist_snowball_factor"]

        # Combined "match pace expectation": weighted avg of both teams' historical durations
        # Teams with more history get more weight
        r_games = rad_feats["rad_hist_games_played"]
        d_games = dir_feats["dire_hist_games_played"]
        total_games = r_games + d_games
        if total_games > 0:
            out["expected_duration"] = (
                rad_feats["rad_hist_avg_duration"] * r_games +
                dir_feats["dire_hist_avg_duration"] * d_games
            ) / total_games
        else:
            out["expected_duration"] = DEFAULT_DURATION

        # "Stall potential": both teams have high pace_score -> mutual late-game tendency
        out["mutual_late_tendency"] = (
            rad_feats["rad_hist_pace_score"] * dir_feats["dire_hist_pace_score"]
        )

        result_rows.append(out)

        # -- Update history AFTER computing features (no leakage!) --
        event_rad = {
            "duration_min": row["duration_min"],
            "won":          row["radiant_win"],
            "gold_at_15":   row["gold_at_15"],    # radiant's gold adv -> from radiant POV
        }
        event_dir = {
            "duration_min": row["duration_min"],
            "won":          not row["radiant_win"],
            "gold_at_15":   -row["gold_at_15"] if row["gold_at_15"] is not None else None,  # flip for dire
        }

        if rid:
            team_hist[rid].append(event_rad)
        if did:
            team_hist[did].append(event_dir)

    result_df = pd.DataFrame(result_rows)
    print(f"Output shape: {result_df.shape}")
    print(f"Columns: {list(result_df.columns)}")
    return result_df


# ---------------------------------------------------------------------------
# Helper: compute features from a team's history window
# ---------------------------------------------------------------------------

def _compute_team_feats(
    history: list[dict],
    window: int,
    late_thresh: float,
    prefix: str,
) -> dict:
    recent = history[-window:] if history else []
    n = len(recent)

    if n == 0:
        return {
            f"{prefix}_hist_avg_duration":     DEFAULT_DURATION,
            f"{prefix}_hist_pace_score":        DEFAULT_PACE,
            f"{prefix}_hist_snowball_factor":   DEFAULT_SNOWBALL,
            f"{prefix}_hist_tempo_std":         DEFAULT_TEMPO_STD,
            f"{prefix}_hist_games_played":      0,
        }

    durations  = [g["duration_min"] for g in recent]
    won_games  = [g for g in recent if g["won"]]

    avg_dur    = float(np.mean(durations))
    tempo_std  = float(np.std(durations)) if n > 1 else DEFAULT_TEMPO_STD
    pace_score = float(np.mean([1.0 if d > late_thresh else 0.0 for d in durations]))

    # Snowball factor: avg gold advantage at min-15 in WON games
    snowball_vals = [
        g["gold_at_15"] for g in won_games
        if g.get("gold_at_15") is not None
    ]
    snowball = float(np.mean(snowball_vals)) if snowball_vals else DEFAULT_SNOWBALL

    return {
        f"{prefix}_hist_avg_duration":     round(avg_dur, 2),
        f"{prefix}_hist_pace_score":        round(pace_score, 4),
        f"{prefix}_hist_snowball_factor":   round(snowball, 1),
        f"{prefix}_hist_tempo_std":         round(tempo_std, 2),
        f"{prefix}_hist_games_played":      n,
    }


# ---------------------------------------------------------------------------
# Validation / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    db = Path(__file__).parent.parent / "dota_data.db"
    feat_df = build_team_history_features(db, window=WINDOW)

    # Merge with duration target to validate
    dur_df = pd.read_csv(Path(__file__).parent.parent / "dota_ml_duration.csv",
                         usecols=["match_id", "duration_min"])
    merged = feat_df.merge(dur_df, on="match_id", how="inner")

    print(f"\nMatches after merge: {len(merged)}")
    print("\nNull counts in hist features:")
    hist_cols = [c for c in feat_df.columns if "hist" in c or c in ["expected_duration","mutual_late_tendency","pace_diff"]]
    print(merged[hist_cols].isnull().sum().to_string())

    print("\nCorrelation with duration_min:")
    from scipy.stats import spearmanr
    corrs = []
    for col in hist_cols:
        r, p = spearmanr(merged[col].fillna(merged[col].median()), merged["duration_min"])
        corrs.append({"feature": col, "spearman_r": round(r, 3), "p": round(p, 4)})
    corr_df = pd.DataFrame(corrs).sort_values("spearman_r", ascending=False)
    print(corr_df.to_string(index=False))

    print("\nSample rows (last 5):")
    print(merged[["match_id","duration_min"] + hist_cols[:6]].tail(5).to_string(index=False))

    # Coverage: how many matches have >0 games played for at least one team
    has_history = ((merged["rad_hist_games_played"] > 0) |
                   (merged["dire_hist_games_played"] > 0)).sum()
    print(f"\nMatches with at least 1 team having history: {has_history}/{len(merged)} "
          f"({100*has_history/len(merged):.1f}%)")
