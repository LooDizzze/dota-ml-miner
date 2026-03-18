"""
processor.py v2 — Pre-Match Features Only (без Target Leakage)

Извлекает ТОЛЬКО данные, известные на момент окончания драфта:
  - радиант победил или нет (целевая переменная)
  - недавний винрейт команды (последние 20 матчей до текущего)
  - исторический винрейт каждого игрока на конкретном пикнутом герое

УДАЛЕНО: GPM, золото, опыт, урон и любые другие внутриигровые метрики.
"""

import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH  = Path(__file__).parent / "dota_data.db"
OUT_CSV  = Path(__file__).parent / "dota_signals_v2_prematch.csv"

# Сколько последних матчей команды учитываем при подсчёте винрейта
TEAM_WINDOW = 20

# Дефолтный винрейт игрока на герое, если он ни разу не играл на нём раньше
DEFAULT_HERO_WINRATE = 0.5


# ---------------------------------------------------------------------------
# Шаг 1: Загрузка всех матчей из базы
# ---------------------------------------------------------------------------

def load_matches(db_path: Path) -> list[dict]:
    """
    Читает все записи из таблицы match_details, парсит JSON и возвращает
    список матчей. Пропускает записи с отсутствующими обязательными полями.
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT raw_data FROM match_details").fetchall()
    conn.close()

    matches = []
    for (raw,) in rows:
        try:
            m = json.loads(raw)
            # Пропускаем матчи без ключевых полей — они не пригодны для обучения
            if m.get("match_id") and m.get("start_time") and m.get("players"):
                matches.append(m)
        except (json.JSONDecodeError, TypeError):
            log.warning("Не удалось распарсить JSON одного из матчей, пропускаем.")

    log.info("Загружено %d корректных матчей из базы.", len(matches))
    return matches


# ---------------------------------------------------------------------------
# Шаг 2: Накопительный расчёт предматчевых признаков
# ---------------------------------------------------------------------------

def compute_features(matches: list[dict]) -> list[dict]:
    """
    Основная функция. Обходит матчи в хронологическом порядке и для каждого
    вычисляет предматчевые признаки на основе ПРОШЛОЙ истории.

    Ключевой принцип anti-leakage:
        1. Вычисляем признаки текущего матча из накопленной истории (прошлое).
        2. Добавляем результат текущего матча в историю (настоящее).
        3. Переходим к следующему матчу.

    Таким образом текущий матч никогда не влияет на собственные признаки,
    и никакие данные из будущего не просачиваются в обучающую выборку.
    """

    # Сортируем по времени — это основа корректности всей логики
    matches_sorted = sorted(matches, key=lambda m: m["start_time"])
    log.info("Матчи отсортированы по start_time. Начинаем вычисление признаков...")

    # История команд: team_id -> список булевых (True = победа) в хронологическом порядке
    # Мы храним полную историю и берём срез [-TEAM_WINDOW:] при вычислении
    team_history: defaultdict[int, list[bool]] = defaultdict(list)

    # Статистика игрока на конкретном герое: (account_id, hero_id) -> [wins, games]
    # Используем lambda чтобы каждый ключ получил свой независимый список
    player_hero_stats: defaultdict[tuple, list] = defaultdict(lambda: [0, 0])

    rows_out = []

    for idx, match in enumerate(matches_sorted):
        if idx % 500 == 0 and idx > 0:
            log.info("[%d/%d] Обрабатываем матчи...", idx, len(matches_sorted))

        match_id        = match.get("match_id")
        start_time      = match.get("start_time")
        patch_version   = match.get("patch")
        radiant_win     = int(bool(match.get("radiant_win")))
        radiant_team_id = match.get("radiant_team_id")
        dire_team_id    = match.get("dire_team_id")
        players         = match.get("players") or []

        # -- Разделяем игроков на команды и сортируем по слоту (позиции 1-5) --
        radiant_players = sorted(
            [p for p in players if p.get("player_slot", 0) < 128],
            key=lambda p: p.get("player_slot", 0),
        )
        dire_players = sorted(
            [p for p in players if p.get("player_slot", 0) >= 128],
            key=lambda p: p.get("player_slot", 0),
        )

        # -------------------------------------------------------------------
        # Признак A: Недавний винрейт команды (последние TEAM_WINDOW матчей)
        # -------------------------------------------------------------------

        def _team_winrate(team_id) -> float | None:
            """
            Возвращает долю побед команды за последние TEAM_WINDOW матчей.
            Если команда ещё не играла — возвращает None (будем заполнять
            медианой или дропать при обучении).
            """
            if not team_id:
                return None
            history = team_history[team_id]
            recent = history[-TEAM_WINDOW:]
            if not recent:
                return None
            return round(sum(recent) / len(recent), 4)

        radiant_recent_winrate = _team_winrate(radiant_team_id)
        dire_recent_winrate    = _team_winrate(dire_team_id)

        # -------------------------------------------------------------------
        # Признак B: Исторический винрейт каждого игрока на пикнутом герое
        # -------------------------------------------------------------------

        def _hero_winrate(account_id, hero_id) -> float:
            """
            Возвращает винрейт игрока на конкретном герое по всем прошлым
            матчам. Если игрок ни разу не брал этого героя — возвращает 0.5
            (нейтральный приор «50/50»).
            """
            if not account_id or not hero_id:
                return DEFAULT_HERO_WINRATE
            wins, games = player_hero_stats[(account_id, hero_id)]
            if games == 0:
                return DEFAULT_HERO_WINRATE
            return round(wins / games, 4)

        # Собираем винрейты для позиций r1..r5 и d1..d5
        # Если в команде вдруг меньше 5 игроков — заполняем дефолтом
        hero_winrates = {}
        for i, p in enumerate(radiant_players[:5], start=1):
            hero_winrates[f"r{i}_hero_winrate"] = _hero_winrate(
                p.get("account_id"), p.get("hero_id")
            )
        for i in range(len(radiant_players) + 1, 6):
            hero_winrates[f"r{i}_hero_winrate"] = DEFAULT_HERO_WINRATE

        for i, p in enumerate(dire_players[:5], start=1):
            hero_winrates[f"d{i}_hero_winrate"] = _hero_winrate(
                p.get("account_id"), p.get("hero_id")
            )
        for i in range(len(dire_players) + 1, 6):
            hero_winrates[f"d{i}_hero_winrate"] = DEFAULT_HERO_WINRATE

        # -------------------------------------------------------------------
        # Формируем итоговую строку (только предматчевые данные!)
        # -------------------------------------------------------------------
        row = {
            "match_id":                match_id,
            "start_time":              start_time,
            "patch_version":           patch_version,
            "radiant_win":             radiant_win,          # целевая переменная
            "radiant_recent_winrate":  radiant_recent_winrate,
            "dire_recent_winrate":     dire_recent_winrate,
            **hero_winrates,
        }
        rows_out.append(row)

        # -------------------------------------------------------------------
        # Обновляем историю ПОСЛЕ вычисления признаков — это критически важно!
        # -------------------------------------------------------------------

        # Обновляем историю команд
        if radiant_team_id:
            team_history[radiant_team_id].append(radiant_win == 1)
        if dire_team_id:
            team_history[dire_team_id].append(radiant_win == 0)  # Dire победила = radiant_win == 0

        # Обновляем статистику каждого игрока на его герое
        for p in players:
            acc_id  = p.get("account_id")
            hero_id = p.get("hero_id")
            if not acc_id or not hero_id:
                continue

            is_radiant = p.get("player_slot", 0) < 128
            won = (radiant_win == 1 and is_radiant) or (radiant_win == 0 and not is_radiant)

            stats = player_hero_stats[(acc_id, hero_id)]
            stats[0] += int(won)  # wins
            stats[1] += 1         # total games

    log.info("Вычисление завершено: %d строк готово.", len(rows_out))
    return rows_out


# ---------------------------------------------------------------------------
# Шаг 3: Сохранение датасета в CSV
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict], out_path: Path) -> None:
    """
    Преобразует список строк в DataFrame и сохраняет в CSV.
    Использует pandas для корректной обработки None-значений (сохраняются как пустые ячейки).
    """
    if not rows:
        log.warning("Нет данных для сохранения.")
        return

    df = pd.DataFrame(rows)

    # Краткая статистика по пропускам — полезно перед обучением модели
    null_counts = df.isnull().sum()
    if null_counts.any():
        log.info("Пропуски в данных (None):\n%s", null_counts[null_counts > 0].to_string())

    df.to_csv(out_path, index=False, encoding="utf-8")
    log.info(
        "CSV сохранён: %s  (%d строк, %d колонок)",
        out_path, len(df), len(df.columns),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not DB_PATH.exists():
        log.error("База данных не найдена: %s — сначала запустите miner.py", DB_PATH)
        return

    matches = load_matches(DB_PATH)
    if not matches:
        log.error("В базе нет матчей для обработки.")
        return

    rows = compute_features(matches)
    save_csv(rows, OUT_CSV)
    log.info("Готово. Выходной файл: %s", OUT_CSV)


if __name__ == "__main__":
    main()
