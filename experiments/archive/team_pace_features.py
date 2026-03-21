# -*- coding: utf-8 -*-
"""
team_pace_features.py
=====================
ЗАДАЧА 1: Leak-free team historical pace features для processor_duration.py

Архитектурная заметка по SQL vs Python:
  match_details хранит данные как JSON-блобы (raw_data TEXT).
  Нет нормализованной player_matches таблицы, поэтому чистый SQL JOIN
  невозможен без предварительной денормализации.

  Правильная стратегия:
    1. Денормализовать JSON → временная таблица в SQLite (один раз за запуск)
    2. Выполнить SQL-запрос с WHERE st_prev < st_current (leak-free окно)
    3. Использовать результат в processor_duration.py

Функция compute_team_pace_features() — готова к интеграции в compute_features().
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WINDOW      = 20     # последние N матчей команды
LATE_THRESH = 40.0   # минут — порог "поздней игры"
MIN_GAMES   = 3      # минимум матчей для надёжной оценки

# Глобальные дефолты для cold start (используются когда history < MIN_GAMES)
# Значения = медианные по датасету (441 матч, mean=40.3, std=10.6)
COLD_AVG_DURATION  = 40.3
COLD_LATE_TENDENCY = 0.47   # доля матчей >40 мин в датасете
COLD_TEMPO_STD     = 10.6   # глобальный std


# ---------------------------------------------------------------------------
# Шаг 1: Денормализация JSON → SQLite temp table
# ---------------------------------------------------------------------------

DENORM_SQL = """
CREATE TEMPORARY TABLE IF NOT EXISTS match_flat (
    match_id        INTEGER PRIMARY KEY,
    start_time      INTEGER,
    duration_sec    INTEGER,
    duration_min    REAL,
    radiant_team_id INTEGER,
    dire_team_id    INTEGER
);
"""

INSERT_SQL = """
INSERT OR IGNORE INTO match_flat
    (match_id, start_time, duration_sec, duration_min, radiant_team_id, dire_team_id)
VALUES (?, ?, ?, ?, ?, ?)
"""

def _build_flat_table(conn: sqlite3.Connection) -> int:
    """
    Парсит raw_data JSON и заполняет temp-таблицу match_flat.
    Возвращает количество вставленных строк.
    """
    conn.execute(DENORM_SQL)
    rows = conn.execute("SELECT raw_data FROM match_details").fetchall()

    batch = []
    for (raw,) in rows:
        try:
            m = json.loads(raw)
            mid  = m.get("match_id")
            st   = m.get("start_time")
            dur  = m.get("duration", 0)
            rid  = m.get("radiant_team_id")
            did  = m.get("dire_team_id")
            if mid and st and dur > 0:
                batch.append((mid, st, dur, round(dur / 60, 2), rid, did))
        except Exception:
            continue

    conn.executemany(INSERT_SQL, batch)
    conn.commit()
    return len(batch)


# ---------------------------------------------------------------------------
# Шаг 2: SQL-запрос (leak-free) для одной команды
# ---------------------------------------------------------------------------

HISTORY_SQL = """
SELECT duration_min
FROM match_flat
WHERE (radiant_team_id = :team_id OR dire_team_id = :team_id)
  AND start_time < :current_start_time
ORDER BY start_time DESC
LIMIT :window
"""

# Объяснение leak-free условия:
#   start_time < :current_start_time  ← строго меньше, не <=
#   (если два матча стартуют в одну секунду — консервативно исключаем оба)
#   ORDER BY start_time DESC LIMIT window ← берём последние N до текущего момента

def _fetch_team_history_sql(
    conn: sqlite3.Connection,
    team_id: Optional[int],
    current_start_time: int,
    window: int = WINDOW,
) -> list[float]:
    """
    Возвращает список duration_min последних N матчей команды
    до current_start_time (без утечки данных).
    """
    if not team_id:
        return []
    rows = conn.execute(HISTORY_SQL, {
        "team_id":             team_id,
        "current_start_time":  current_start_time,
        "window":              window,
    }).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Шаг 3: Вычисление 3 фич из истории
# ---------------------------------------------------------------------------

def _compute_pace_feats(
    durations: list[float],
    prefix: str,
    late_thresh: float = LATE_THRESH,
    min_games: int = MIN_GAMES,
) -> dict:
    """
    Вычисляет 3 фичи темпа из списка прошлых длительностей.

    Cold start (< min_games матчей): возвращает глобальные медианные значения
    с флагом {prefix}_cold_start=1, чтобы CatBoost мог отличить
    "реальное значение" от "заглушки".
    """
    n = len(durations)

    if n < min_games:
        return {
            f"{prefix}_avg_duration":    COLD_AVG_DURATION,
            f"{prefix}_late_tendency":   COLD_LATE_TENDENCY,
            f"{prefix}_tempo_std":       COLD_TEMPO_STD,
            f"{prefix}_games_count":     n,
            f"{prefix}_cold_start":      1,
        }

    arr = np.array(durations)
    return {
        f"{prefix}_avg_duration":    round(float(arr.mean()), 2),
        f"{prefix}_late_tendency":   round(float((arr > late_thresh).mean()), 4),
        f"{prefix}_tempo_std":       round(float(arr.std(ddof=1)), 2) if n > 1 else COLD_TEMPO_STD,
        f"{prefix}_games_count":     n,
        f"{prefix}_cold_start":      0,
    }


# ---------------------------------------------------------------------------
# Шаг 4: Основная функция для интеграции в processor_duration.py
# ---------------------------------------------------------------------------

def compute_team_pace_features(
    match: dict,
    conn: sqlite3.Connection,
    window: int = WINDOW,
) -> dict:
    """
    Главная функция — вызывается внутри цикла compute_features() для каждого матча.

    ВАЖНО: вызывать ДО обновления истории (строка с team_history.append),
    иначе текущий матч попадёт в свою же историю.

    Args:
        match:  сырой dict матча (из JSON)
        conn:   открытое SQLite-соединение с уже построенной temp-таблицей
        window: окно истории

    Returns:
        dict с 10 новыми фичами (5 для каждой стороны)
    """
    rid  = match.get("radiant_team_id")
    did  = match.get("dire_team_id")
    st   = match.get("start_time")

    rad_hist  = _fetch_team_history_sql(conn, rid, st, window)
    dire_hist = _fetch_team_history_sql(conn, did, st, window)

    rad_feats  = _compute_pace_feats(rad_hist,  prefix="rad_team")
    dire_feats = _compute_pace_feats(dire_hist, prefix="dire_team")

    # Interaction фичи
    interactions = {
        "team_avg_duration_diff":   rad_feats["rad_team_avg_duration"] - dire_feats["dire_team_avg_duration"],
        "team_late_tendency_diff":  rad_feats["rad_team_late_tendency"] - dire_feats["dire_team_late_tendency"],
        "team_expected_duration":   (rad_feats["rad_team_avg_duration"] + dire_feats["dire_team_avg_duration"]) / 2,
        "team_max_late_tendency":   max(rad_feats["rad_team_late_tendency"], dire_feats["dire_team_late_tendency"]),
    }

    return {**rad_feats, **dire_feats, **interactions}


# ---------------------------------------------------------------------------
# Шаг 5: Инициализация соединения (вызвать один раз в начале processor_duration.py)
# ---------------------------------------------------------------------------

def init_team_history_connection(db_path: str | Path) -> sqlite3.Connection:
    """
    Открывает соединение, создаёт и заполняет temp-таблицу match_flat.
    Возвращает conn — передавать в compute_team_pace_features() для каждого матча.

    Пример использования в processor_duration.py:

        from experiments.team_pace_features import (
            init_team_history_connection,
            compute_team_pace_features,
        )

        # В main() перед основным циклом:
        pace_conn = init_team_history_connection(DB_PATH)

        # Внутри цикла compute_features(), перед обновлением team_history:
        pace_feats = compute_team_pace_features(match, pace_conn)
        row.update(pace_feats)

        # После цикла:
        pace_conn.close()
    """
    conn = sqlite3.connect(db_path)
    n = _build_flat_table(conn)
    print(f"[team_pace] Flat table built: {n} matches indexed for history queries.")
    return conn


# ---------------------------------------------------------------------------
# Validation / standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    db = Path(__file__).parent.parent / "dota_data.db"
    dur_csv = Path(__file__).parent.parent / "dota_ml_duration.csv"

    print("=== team_pace_features.py standalone test ===\n")

    conn = init_team_history_connection(db)

    # Читаем все матчи, считаем фичи, меряем корреляцию с таргетом
    rows_all = conn.execute("SELECT raw_data FROM match_details").fetchall()
    matches = []
    for (raw,) in rows_all:
        try:
            m = json.loads(raw)
            if m.get("start_time") and m.get("duration", 0) > 0:
                matches.append(m)
        except Exception:
            continue
    matches.sort(key=lambda m: m["start_time"])

    results = []
    for m in matches:
        feats = compute_team_pace_features(m, conn)
        feats["match_id"]    = m["match_id"]
        feats["duration_min"] = round(m["duration"] / 60, 2)
        results.append(feats)

    conn.close()

    df = pd.DataFrame(results)
    print(f"Output shape: {df.shape}")
    print(f"Cold start (rad): {df['rad_team_cold_start'].sum()}/{len(df)}")
    print(f"Cold start (dire): {df['dire_team_cold_start'].sum()}/{len(df)}")

    from scipy.stats import spearmanr
    feat_cols = [c for c in df.columns if c not in ("match_id","duration_min")]
    print("\nCorrelation with duration_min:")
    corrs = []
    for col in feat_cols:
        r, p = spearmanr(df[col], df["duration_min"])
        corrs.append({"feature": col, "spearman_r": round(r, 3), "p": round(p, 4)})
    corr_df = pd.DataFrame(corrs).sort_values("spearman_r", ascending=False)
    print(corr_df.to_string(index=False))

    # Быстрый CatBoost тест: history feats only vs baseline
    from catboost import CatBoostRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error

    base_df = pd.read_csv(dur_csv).sort_values("start_time").reset_index(drop=True)
    drop_c  = ["match_id","start_time","patch_version","duration_min","radiant_win"]
    X_base  = base_df.drop(columns=[c for c in drop_c if c in base_df.columns])
    X_base  = X_base.fillna(X_base.median(numeric_only=True))
    y       = base_df["duration_min"]

    merged = base_df[["match_id"]].merge(df.drop(columns=["duration_min"]), on="match_id", how="left")
    hist_feats = [c for c in df.columns if c not in ("match_id","duration_min")]
    X_hist = merged[hist_feats].fillna({
        c: COLD_AVG_DURATION if "avg_duration" in c or "expected" in c
        else COLD_LATE_TENDENCY if "tendency" in c
        else COLD_TEMPO_STD if "tempo_std" in c
        else 0
        for c in hist_feats
    })

    X_combined = pd.concat([X_base.reset_index(drop=True),
                             X_hist.reset_index(drop=True)], axis=1)

    params = dict(iterations=1000, learning_rate=0.01, depth=3, l2_leaf_reg=15,
                  loss_function="RMSE", early_stopping_rounds=100, random_seed=42, verbose=False)

    def quick_cv(X, y, label):
        rmses = []
        for tr, te in TimeSeriesSplit(n_splits=4).split(X):
            m = CatBoostRegressor(**params)
            m.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[te], y.iloc[te]), use_best_model=True)
            rmses.append(np.sqrt(mean_squared_error(y.iloc[te], m.predict(X.iloc[te]))))
        print(f"  [{label}] folds={[round(r,2) for r in rmses]}  MEAN={np.mean(rmses):.3f}")
        return float(np.mean(rmses))

    print("\n=== Quick CatBoost CV comparison ===")
    r1 = quick_cv(X_base,     y, "BASELINE (tags only)")
    r2 = quick_cv(X_combined, y, "TAGS + TEAM HISTORY ")
    print(f"\n  Delta RMSE = {r1 - r2:+.3f} min  ({'IMPROVED' if r1 > r2 else 'DEGRADED'})")
