"""
predict_api.py — FastAPI microservice for Dota 2 match outcome prediction

POST /predict
  Input:  { radiant_hero_ids: [5 ints], dire_hero_ids: [5 ints],
            radiant_team_id?: int, dire_team_id?: int }
  Output: { radiant_win_prob: float, dire_win_prob: float,
            features_used: dict, has_team_data: bool }

GET /health
  Returns model status and feature count.

Run:
  uvicorn predict_api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from catboost import CatBoostClassifier
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR       = Path(__file__).parent
MODEL_PATH     = BASE_DIR / "dota_model.cbm"
HERO_TAGS_PATH = BASE_DIR / "hero_tags_full.json"
DB_PATH        = BASE_DIR / "dota_data.db"

TEAM_WINDOW = 20

SYNERGY_PAIRS = [
    ("initiation",  "burst_damage"),
    ("hard_save",   "late_game_scaling"),
    ("waveclear",   "elusive"),
]

# ---------------------------------------------------------------------------
# Startup: load model, tags, histories
# ---------------------------------------------------------------------------

log.info("Loading CatBoost model: %s", MODEL_PATH)
model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))
FEATURE_NAMES: list[str] = list(model.feature_names_)
log.info("Model loaded. Features: %d  ->  %s", len(FEATURE_NAMES), FEATURE_NAMES)

# Hero tags
log.info("Loading hero_tags_full.json...")
_tags_raw = json.loads(HERO_TAGS_PATH.read_text(encoding="utf-8"))
hero_tags: dict[str, set[str]] = {}
_all_tags: set[str] = set()
for entry in _tags_raw:
    name = entry.get("hero", "").lower()
    if not name:
        continue
    tags: set[str] = set()
    for ab in entry.get("abilities", []):
        for t in ab.get("tags", []):
            tags.add(t)
            _all_tags.add(t)
    hero_tags[name] = tags
ALL_TAGS: list[str] = sorted(_all_tags)
log.info("Tags loaded: %d heroes, %d unique tags", len(hero_tags), len(ALL_TAGS))


def _fetch_hero_id_map() -> dict[int, str]:
    try:
        resp = requests.get("https://api.opendota.com/api/constants/heroes", timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        return {v["id"]: v["localized_name"].lower() for v in raw.values() if "id" in v}
    except Exception as e:
        log.warning("Could not fetch hero id map: %s", e)
        return {}


log.info("Fetching hero_id -> name map from OpenDota...")
hero_id_map: dict[int, str] = _fetch_hero_id_map()
log.info("Hero map: %d heroes", len(hero_id_map))

# Team history and hero global stats from DB
team_history: dict[int, list[bool]] = defaultdict(list)
hero_global_stats: dict[int, list[int]] = defaultdict(lambda: [0, 0])


def _build_histories() -> None:
    if not DB_PATH.exists():
        log.warning("DB not found, team winrate features will use defaults.")
        return
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT raw_data FROM match_details").fetchall()
    conn.close()

    matches = []
    for (raw,) in rows:
        try:
            m = json.loads(raw)
            if m.get("start_time") and m.get("players"):
                matches.append(m)
        except Exception:
            continue
    matches.sort(key=lambda m: m["start_time"])

    for m in matches:
        radiant_win     = bool(m.get("radiant_win"))
        radiant_team_id = m.get("radiant_team_id")
        dire_team_id    = m.get("dire_team_id")
        players         = m.get("players", [])

        if radiant_team_id:
            team_history[radiant_team_id].append(radiant_win)
        if dire_team_id:
            team_history[dire_team_id].append(not radiant_win)

        for p in players:
            hero_id = p.get("hero_id")
            if not hero_id:
                continue
            is_radiant = p.get("player_slot", 0) < 128
            won = (radiant_win and is_radiant) or (not radiant_win and not is_radiant)
            g = hero_global_stats[hero_id]
            g[0] += int(won)
            g[1] += 1

    log.info("Histories built: %d teams, %d heroes", len(team_history), len(hero_global_stats))


_build_histories()

# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def _team_winrate(team_id: Optional[int]) -> Optional[float]:
    if not team_id or team_id not in team_history:
        return None
    recent = team_history[team_id][-TEAM_WINDOW:]
    if not recent:
        return None
    return round(sum(recent) / len(recent), 4)


def _tag_counts(hero_ids: list[int], prefix: str) -> dict[str, int]:
    counts = {f"{prefix}_{tag}_count": 0 for tag in ALL_TAGS}
    for hid in hero_ids:
        name = hero_id_map.get(hid, "")
        for tag in hero_tags.get(name, set()):
            key = f"{prefix}_{tag}_count"
            if key in counts:
                counts[key] += 1
    return counts


def _synergy_score(hero_ids: list[int]) -> int:
    team_tags: set[str] = set()
    for hid in hero_ids:
        name = hero_id_map.get(hid, "")
        team_tags |= hero_tags.get(name, set())
    return sum(1 for a, b in SYNERGY_PAIRS if a in team_tags and b in team_tags)


def _meta_score(hero_ids: list[int]) -> float:
    rates = []
    for hid in hero_ids:
        s = hero_global_stats.get(hid, [0, 0])
        rates.append(round(s[0] / s[1], 4) if s[1] > 0 else 0.5)
    while len(rates) < 5:
        rates.append(0.5)
    return round(sum(rates) / len(rates), 4)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Dota2 ML Predictor", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    radiant_hero_ids: list[int]   # up to 5
    dire_hero_ids:    list[int]   # up to 5
    radiant_team_id:  Optional[int] = None
    dire_team_id:     Optional[int] = None


@app.post("/predict")
def predict(req: PredictRequest):
    rad_ids  = req.radiant_hero_ids[:5]
    dire_ids = req.dire_hero_ids[:5]

    rad_winrate  = _team_winrate(req.radiant_team_id)
    dire_winrate = _team_winrate(req.dire_team_id)
    has_team_data = rad_winrate is not None or dire_winrate is not None

    winrate_adv = None
    if rad_winrate is not None and dire_winrate is not None:
        winrate_adv = round(rad_winrate - dire_winrate, 4)

    rad_tags  = _tag_counts(rad_ids,  "radiant")
    dire_tags = _tag_counts(dire_ids, "dire")

    all_features: dict = {
        "radiant_recent_winrate":    rad_winrate,
        "dire_recent_winrate":       dire_winrate,
        "radiant_winrate_advantage": winrate_adv,
        "radiant_synergy_score":     _synergy_score(rad_ids),
        "dire_synergy_score":        _synergy_score(dire_ids),
        "radiant_meta_score":        _meta_score(rad_ids),
        "dire_meta_score":           _meta_score(dire_ids),
        "radiant_first_pick":        0,
        # player hero winrates default to 0.5 (unknown in draft mode)
        **{f"r{i}_hero_winrate": 0.5 for i in range(1, 6)},
        **{f"d{i}_hero_winrate": 0.5 for i in range(1, 6)},
        **rad_tags,
        **dire_tags,
    }

    # Build feature vector in exact model order; None -> 0.5 (≈ median for winrate features)
    row = {
        feat: (all_features[feat] if all_features.get(feat) is not None else 0.5)
        for feat in FEATURE_NAMES
    }

    X     = pd.DataFrame([row])
    proba = model.predict_proba(X)[0]

    return {
        "radiant_win_prob": round(float(proba[1]), 4),
        "dire_win_prob":    round(float(proba[0]), 4),
        "has_team_data":    has_team_data,
        "features_used":    row,
    }


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "features":      len(FEATURE_NAMES),
        "teams_tracked": len(team_history),
        "heroes_tracked": len(hero_global_stats),
    }
