"""
processor.py — Extract signals from raw match JSON stored in dota_data.db
and write a flat CSV ready for ML training.

Architecture
------------
Each "Signal" is an isolated function:

    extract_<signal_name>(match: dict) -> dict

It receives the parsed match JSON and returns a flat dict of feature columns.
`process_match()` merges all signal dicts into one row.
To add a new signal: write extract_mySignal(), add it to SIGNAL_EXTRACTORS.
"""

import csv
import json
import logging
import math
import sqlite3
import statistics
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "dota_data.db"
OUT_CSV = Path(__file__).parent / "dota_signals_v1.csv"

# Gold-graph snapshots are stored at 1-minute intervals in OpenDota JSON.
MINUTE_15 = 15
MINUTE_25 = 25
GOLD_LEAD_THRESHOLD = 5_000  # for Map Pressure Retention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(value: Any, default=None):
    """Return value if truthy (but allow 0), else default."""
    return value if value is not None else default


def _split_teams(players: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split player list into (radiant, dire) by player_slot."""
    radiant = [p for p in players if p.get("player_slot", 0) < 128]
    dire    = [p for p in players if p.get("player_slot", 0) >= 128]
    return radiant, dire


def _get_minute(series: list, minute: int, default=None):
    """Safely index a per-minute time-series list."""
    if series and len(series) > minute:
        return series[minute]
    return default


# ---------------------------------------------------------------------------
# Signal 1 — Draft Integrity
# ---------------------------------------------------------------------------

def extract_draft_integrity(match: dict) -> dict:
    """
    Columns:
        match_id, league_id, start_time, patch_version,
        radiant_hero_ids, dire_hero_ids,
        radiant_account_ids, dire_account_ids,
        radiant_win (target label)
    """
    players = match.get("players") or []
    radiant, dire = _split_teams(players)

    return {
        "match_id":           match.get("match_id"),
        "league_id":          match.get("leagueid"),
        "start_time":         match.get("start_time"),
        "patch_version":      match.get("patch"),
        "radiant_hero_ids":   str([p.get("hero_id") for p in radiant]),
        "dire_hero_ids":      str([p.get("hero_id") for p in dire]),
        "radiant_account_ids": str([p.get("account_id") for p in radiant]),
        "dire_account_ids":   str([p.get("account_id") for p in dire]),
        "radiant_win":        int(bool(match.get("radiant_win"))),
    }


# ---------------------------------------------------------------------------
# Signal 2 — Technicals
# ---------------------------------------------------------------------------

def _early_economy_delta(match: dict) -> dict:
    """
    early_gold_delta_15  : radiant_gold - dire_gold at minute 15
    early_xp_delta_15    : radiant_xp   - dire_xp   at minute 15
    """
    gold_adv = match.get("radiant_gold_adv") or []
    xp_adv   = match.get("radiant_xp_adv") or []

    return {
        "early_gold_delta_15": _get_minute(gold_adv, MINUTE_15),
        "early_xp_delta_15":   _get_minute(xp_adv,  MINUTE_15),
    }


def _core_execution_synergy(players: list[dict], prefix: str) -> dict:
    """
    (Sum of damage dealt by top-3 heroes by net-worth) / (Their total net-worth)
    Returns NaN if data is missing.
    """
    key = f"{prefix}_core_execution_synergy"
    if not players:
        return {key: None}

    sorted_by_nw = sorted(
        players,
        key=lambda p: _safe(p.get("net_worth"), 0),
        reverse=True,
    )
    top3 = sorted_by_nw[:3]

    total_nw  = sum(_safe(p.get("net_worth"), 0) for p in top3)
    total_dmg = sum(_safe(p.get("hero_damage"), 0) for p in top3)

    value = (total_dmg / total_nw) if total_nw > 0 else None
    return {key: value}


def _map_pressure_retention(match: dict) -> dict:
    """
    For the team that had >5000 gold lead at minute 25,
    compute std-dev of the gold advantage series from minute 25 onwards.
    Low std-dev = stable; high = volatile.
    """
    gold_adv = match.get("radiant_gold_adv") or []
    adv_at_25 = _get_minute(gold_adv, MINUTE_25)

    result: dict = {
        "pressure_team":           None,   # "radiant" | "dire" | None
        "pressure_retention_std":  None,
    }

    if adv_at_25 is None:
        return result

    if abs(adv_at_25) <= GOLD_LEAD_THRESHOLD:
        return result  # no significant lead at 25 min

    tail = gold_adv[MINUTE_25:]
    if len(tail) < 2:
        return result

    if adv_at_25 > 0:
        result["pressure_team"] = "radiant"
        series = tail                       # positive = radiant ahead
    else:
        result["pressure_team"] = "dire"
        series = [-v for v in tail]         # flip so "bigger = better" for dire

    result["pressure_retention_std"] = statistics.stdev(series)
    return result


def extract_technicals(match: dict) -> dict:
    players = match.get("players") or []
    radiant, dire = _split_teams(players)

    out: dict = {}
    out.update(_early_economy_delta(match))
    out.update(_core_execution_synergy(radiant, "radiant"))
    out.update(_core_execution_synergy(dire, "dire"))
    out.update(_map_pressure_retention(match))
    return out


# ---------------------------------------------------------------------------
# Signal 3 — Form & Psychology (raw values; rolling volatility via Pandas next)
# ---------------------------------------------------------------------------

def extract_form_psychology(match: dict) -> dict:
    """
    Collect gold_per_min for Pos 1 (carry) and Pos 2 (mid) from each team.
    Lane role: lane_role == 1 → safe (pos1 candidate); 2 → mid; etc.
    We take the highest-GPM player among lane_role==1 as Pos1 proxy,
    and highest-GPM among lane_role==2 as Pos2 proxy, per team.
    """
    players = match.get("players") or []
    radiant, dire = _split_teams(players)

    def _gpm_by_role(team: list[dict], role: int) -> int | None:
        candidates = [p for p in team if p.get("lane_role") == role]
        if not candidates:
            return None
        return max(_safe(p.get("gold_per_min"), 0) for p in candidates)

    return {
        "radiant_pos1_gpm": _gpm_by_role(radiant, 1),
        "radiant_pos2_gpm": _gpm_by_role(radiant, 2),
        "dire_pos1_gpm":    _gpm_by_role(dire,    1),
        "dire_pos2_gpm":    _gpm_by_role(dire,    2),
    }


# ---------------------------------------------------------------------------
# Signal registry — add new signals here
# ---------------------------------------------------------------------------

SIGNAL_EXTRACTORS = [
    extract_draft_integrity,
    extract_technicals,
    extract_form_psychology,
]


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_match(raw_json: str) -> dict | None:
    """Parse one raw JSON string and return a merged feature row."""
    try:
        match = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        log.warning("JSON parse error: %s", exc)
        return None

    row: dict = {}
    for extractor in SIGNAL_EXTRACTORS:
        try:
            row.update(extractor(match))
        except Exception as exc:  # noqa: BLE001
            log.warning("Extractor %s failed: %s", extractor.__name__, exc)

    return row


def load_and_process(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    rows_db = conn.execute("SELECT match_id, raw_data FROM match_details").fetchall()
    conn.close()

    total = len(rows_db)
    log.info("Processing %d matches from DB...", total)

    results = []
    skipped = 0
    for idx, (match_id, raw_data) in enumerate(rows_db, start=1):
        if idx % 100 == 0 or idx == total:
            log.info("[%d/%d] Processing match %d", idx, total, match_id)
        row = process_match(raw_data)
        if row:
            results.append(row)
        else:
            skipped += 1

    log.info("Done. Processed: %d, Skipped: %d", len(results), skipped)
    return results


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        log.warning("No rows to write.")
        return

    # Preserve column order: all keys from first row, then any extras from others
    fieldnames: list[str] = list(rows[0].keys())
    for row in rows[1:]:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log.info("CSV written: %s  (%d rows, %d columns)", out_path, len(rows), len(fieldnames))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not DB_PATH.exists():
        log.error("Database not found: %s — run miner.py first.", DB_PATH)
        return

    rows = load_and_process(DB_PATH)
    write_csv(rows, OUT_CSV)
    log.info("All done. Output: %s", OUT_CSV)


if __name__ == "__main__":
    main()
