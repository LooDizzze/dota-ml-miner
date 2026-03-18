# Sample Data — dota-ml-miner

This file shows real examples of what the pipeline collects.

---

## What the project does

1. **miner.py** — fetches pro league matches from OpenDota API → stores raw JSON in SQLite (`match_details` table)
2. **player_miner.py** — for every player found in those matches, fetches their last year of pro results → stores in `player_stats_pro` table
3. **processor.py** — reads raw match JSON, extracts ML features, writes `dota_signals_v1.csv`

---

## Sample: 1 Match (from `match_details`)

```json
{
  "match_id": 8608974467,
  "leagueid": 18988,
  "start_time": 1765968902,
  "patch": 59,
  "radiant_win": true,
  "duration": 3041,
  "radiant_gold_adv_first_5_min": [373, 468, 685, 471, 981],
  "radiant_xp_adv_first_5_min": [99, 48, 315, -112, 240],
  "players_sample_2of10": [
    {
      "hero_id": 102,
      "player_slot": 0,
      "account_id": 1044002267,
      "kills": 15,
      "deaths": 2,
      "assists": 12,
      "gold_per_min": 762,
      "xp_per_min": 979,
      "hero_damage": 44087,
      "net_worth": 36055,
      "lane_role": 1
    },
    {
      "hero_id": 129,
      "player_slot": 1,
      "account_id": 56351509,
      "kills": 6,
      "deaths": 5,
      "assists": 21,
      "gold_per_min": 534,
      "xp_per_min": 762,
      "hero_damage": 22621,
      "net_worth": 25621,
      "lane_role": 3
    }
  ]
}
```

Each match has 10 players total (5 radiant, 5 dire). Full JSON includes:
- per-minute gold/xp advantage arrays (full game length)
- all 10 players with full stats: KDA, GPM, XPM, hero_damage, net_worth, lane_role, etc.
- draft info (hero_id per player)
- team IDs, league ID, patch version

---

## Sample: 1 Player (from `player_stats_pro`)

```json
{
  "account_id": 9403474,
  "name": "yamich",
  "total_pro_games_last_year": 404,
  "top3_heroes_by_games": {
    "100": { "games": 106, "wins": 74, "winrate": 0.698 },
    "14":  { "games": 67,  "wins": 37, "winrate": 0.552 },
    "119": { "games": 26,  "wins": 16, "winrate": 0.615 }
  },
  "recent_15_results": ["L","W","L","L","L","W","L","L","L","L","W","W","W","L","L"]
}
```

For every unique player found in the collected matches, we store:
- total pro games in the last year
- hero pool: games + winrate per hero (full pool, not just top 3)
- recent form: last 15 match results as W/L sequence

---

## CSV features extracted by processor.py (per match row)

| Column | Description |
|---|---|
| `match_id` | Unique match identifier |
| `league_id` | Tournament ID |
| `start_time` | Unix timestamp |
| `patch_version` | Game patch number |
| `radiant_hero_ids` | List of 5 hero IDs for Radiant |
| `dire_hero_ids` | List of 5 hero IDs for Dire |
| `radiant_account_ids` | List of 5 player account IDs for Radiant |
| `dire_account_ids` | List of 5 player account IDs for Dire |
| `radiant_win` | **Target label** (1 = Radiant won, 0 = Dire won) |
| `early_gold_delta_15` | Radiant gold advantage at minute 15 |
| `early_xp_delta_15` | Radiant XP advantage at minute 15 |
| `radiant_core_execution_synergy` | Damage/networth ratio for top-3 Radiant cores |
| `dire_core_execution_synergy` | Damage/networth ratio for top-3 Dire cores |
| `pressure_team` | Which team had >5000 gold lead at min 25 |
| `pressure_retention_std` | Std-dev of gold lead after min 25 (stability metric) |
| `radiant_pos1_gpm` | Best GPM among Radiant safe-lane carry |
| `radiant_pos2_gpm` | Best GPM among Radiant mid |
| `dire_pos1_gpm` | Best GPM among Dire carry |
| `dire_pos2_gpm` | Best GPM among Dire mid |

---

## Current DB stats

- Source: OpenDota API (pro leagues only — DreamLeague, ESL One, TI, Majors, etc.)
- No qualifiers, only main events
- ~90 day rolling window of matches
- `dota_data.db` is ~125 MB (excluded from git — too large)
- `dota_signals_v1.csv` is the processed flat feature file (~120 KB)
