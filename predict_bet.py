"""
predict_bet.py — Детектор Лейта: ежедневный скрипт прогнозов на ТБ 38

Использование:
  1. Заполни секцию "ВВОД ДАННЫХ" ниже
  2. Запусти: python predict_bet.py
  3. Если сигнал есть — он автоматически запишется в paper_bets_log.csv
  4. После матча открой CSV и впиши результат в колонку 'result' (W или L)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

MODEL_PATH   = Path(__file__).parent / "dota_model_duration_clf.cbm"
LOG_PATH     = Path(__file__).parent / "paper_bets_log.csv"

LINE           = 38.0
BET_CONFIDENCE = 0.53   # порог: 77 ставок / 22% охват / winrate 62.3% / ROI +15.3%
BET_ODDS       = 1.85

FEATURE_NAMES = [
    "rad_team_avg_duration",
    "rad_team_late_tendency",
    "rad_team_tempo_std",
    "rad_team_games_count",
    "rad_team_cold_start",
    "dire_team_avg_duration",
    "dire_team_late_tendency",
    "dire_team_tempo_std",
    "dire_team_games_count",
    "dire_team_cold_start",
    "team_avg_duration_diff",
    "team_late_tendency_diff",
    "team_expected_duration",
    "team_max_late_tendency",
    "diff_waveclear",
]

# ---------------------------------------------------------------------------
# ВВОД ДАННЫХ — заполни перед каждым матчем
# ---------------------------------------------------------------------------
# Дефолты для команды без истории:
#   avg_duration=40.3, late_tendency=0.47, tempo_std=10.6, games_count=0, cold_start=1

MATCH_ID    = ""                    # ID матча (опционально, можно оставить пустым)
TEAM_RADIANT = "Team Radiant"       # название команды
TEAM_DIRE    = "Team Dire"

match_data = {
    # История Radiant
    "rad_team_avg_duration":   40.3,
    "rad_team_late_tendency":  0.47,
    "rad_team_tempo_std":      10.6,
    "rad_team_games_count":    0,
    "rad_team_cold_start":     1,

    # История Dire
    "dire_team_avg_duration":  40.3,
    "dire_team_late_tendency": 0.47,
    "dire_team_tempo_std":     10.6,
    "dire_team_games_count":   0,
    "dire_team_cold_start":    1,

    # Взаимодействие — считается автоматически ниже
    "team_avg_duration_diff":  0.0,
    "team_late_tendency_diff": 0.0,
    "team_expected_duration":  40.3,
    "team_max_late_tendency":  0.47,

    # Драфт: waveclear Radiant минус waveclear Dire (целое число от -5 до +5)
    "diff_waveclear": 0,
}

# ---------------------------------------------------------------------------
# Авторасчёт интеракционных фич
# ---------------------------------------------------------------------------

rad_avg  = match_data["rad_team_avg_duration"]
dire_avg = match_data["dire_team_avg_duration"]
rad_late = match_data["rad_team_late_tendency"]
dire_late = match_data["dire_team_late_tendency"]
rad_n    = match_data["rad_team_games_count"]
dire_n   = match_data["dire_team_games_count"]

match_data["team_avg_duration_diff"]  = round(rad_avg - dire_avg, 2)
match_data["team_late_tendency_diff"] = round(rad_late - dire_late, 4)
match_data["team_expected_duration"]  = round(
    (rad_avg * max(rad_n, 1) + dire_avg * max(dire_n, 1)) / (max(rad_n, 1) + max(dire_n, 1)), 2
)
match_data["team_max_late_tendency"]  = round(max(rad_late, dire_late), 4)

# ---------------------------------------------------------------------------
# Предсказание
# ---------------------------------------------------------------------------

model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))

X_new = pd.DataFrame([match_data])[FEATURE_NAMES]
proba_over  = float(model.predict_proba(X_new)[0][1])
proba_under = 1.0 - proba_over
ev          = round(BET_ODDS * proba_over - 1, 4)

# ---------------------------------------------------------------------------
# Вывод в консоль
# ---------------------------------------------------------------------------

team_label = f"{TEAM_RADIANT} vs {TEAM_DIRE}"

print()
print("=" * 58)
print(f"  {team_label}")
if MATCH_ID:
    print(f"  Match ID: {MATCH_ID}")
print(f"  Линия: ТБ/ТМ {LINE:.0f} мин  |  Кэф: {BET_ODDS}  |  Порог: {BET_CONFIDENCE:.0%}")
print("=" * 58)
print(f"  Вероятность ТБ {LINE:.0f}: {proba_over*100:.1f}%")
print(f"  Вероятность ТМ {LINE:.0f}: {proba_under*100:.1f}%")
print("-" * 58)

is_bet = proba_over >= BET_CONFIDENCE

if is_bet:
    print(f"  🔥 ЖЕЛЕЗНАЯ СТАВКА: ТБ {LINE:.0f}  (Уверенность: {proba_over*100:.1f}%)")
    print(f"  EV на единицу ставки: {ev:+.4f}  ({ev*100:+.1f}%)")
else:
    print(f"  ⚠️  СКИП  (ТБ {proba_over*100:.1f}% < порога {BET_CONFIDENCE:.0%})")

print("=" * 58)
print()

# ---------------------------------------------------------------------------
# Запись в лог — ТОЛЬКО при наличии сигнала
# ---------------------------------------------------------------------------

if is_bet:
    log_exists = LOG_PATH.exists()

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date", "match_id", "teams",
            "predicted_prob", "bet_type", "odds", "result"
        ])
        if not log_exists:
            writer.writeheader()

        writer.writerow({
            "date":           datetime.now().strftime("%Y-%m-%d %H:%M"),
            "match_id":       MATCH_ID or "—",
            "teams":          team_label,
            "predicted_prob": f"{proba_over*100:.1f}%",
            "bet_type":       f"Over {LINE:.0f}",
            "odds":           BET_ODDS,
            "result":         "",   # заполнишь руками: W или L
        })

    print(f"  ✅ Прогноз записан в {LOG_PATH.name}")
    print()
