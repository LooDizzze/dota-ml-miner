"""
processor.py v4 — Pre-Match Features + Draft Macro-Tags + Synergy + Meta Score (без Target Leakage)

Извлекает ТОЛЬКО данные, известные на момент окончания драфта:
  - radiant_win              — целевая переменная
  - радиант/дир: недавний винрейт команды (последние 20 матчей)
  - r1..r5 / d1..d5         — исторический винрейт игрока на пикнутом герое
  - radiant_{tag}_count      — сумма тегов из hero_tags_full.json по драфту Radiant
  - dire_{tag}_count         — сумма тегов из hero_tags_full.json по драфту Dire

УДАЛЕНО: GPM, золото, опыт, урон и любые другие внутриигровые метрики.
"""

import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH        = Path(__file__).parent / "dota_data.db"
HERO_TAGS_PATH = Path(__file__).parent / "hero_tags_full.json"
OUT_CSV        = Path(__file__).parent / "dota_ml_features_final.csv"

# Сколько последних матчей команды учитываем при подсчёте винрейта
TEAM_WINDOW = 20

# Дефолтный винрейт игрока на герое, если он ни разу не играл на нём раньше
DEFAULT_HERO_WINRATE = 0.5


# ---------------------------------------------------------------------------
# Шаг 0а: Загрузка hero_tags_full.json → маппинг герой -> набор тегов
# ---------------------------------------------------------------------------

def load_hero_tags(path: Path) -> tuple[dict[str, set[str]], list[str]]:
    """
    Читает hero_tags_full.json (ТОЛЬКО чтение — файл священный).

    Возвращает:
      hero_tags  — dict: lowercase_hero_name -> frozenset тегов со всех способностей
      all_tags   — отсортированный список всех уникальных тегов в файле
    """
    if not path.exists():
        log.error("hero_tags_full.json не найден: %s", path)
        return {}, []

    data = json.loads(path.read_text(encoding="utf-8"))

    hero_tags: dict[str, set[str]] = {}
    all_tags: set[str] = set()

    for entry in data:
        hero_name = entry.get("hero", "").lower()
        if not hero_name:
            continue

        tags: set[str] = set()
        for ability in entry.get("abilities", []):
            for tag in ability.get("tags", []):
                tags.add(tag)
                all_tags.add(tag)

        hero_tags[hero_name] = tags

    sorted_tags = sorted(all_tags)
    log.info(
        "hero_tags_full.json загружен: %d героев, %d уникальных тегов.",
        len(hero_tags), len(sorted_tags),
    )
    return hero_tags, sorted_tags


# ---------------------------------------------------------------------------
# Шаг 0б: Загрузка маппинга hero_id -> hero_name из OpenDota API
# ---------------------------------------------------------------------------

def fetch_hero_id_map() -> dict[int, str]:
    """
    Запрашивает /api/constants/heroes и возвращает:
      {hero_id: lowercase_localized_name}

    Пример ответа API:
      {"npc_dota_hero_antimage": {"id": 1, "localized_name": "Anti-Mage"}, ...}
    """
    url = "https://api.opendota.com/api/constants/heroes"
    log.info("Загружаем маппинг hero_id -> name из OpenDota API...")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
    except requests.RequestException as e:
        log.error("Не удалось получить константы героев: %s", e)
        return {}

    mapping: dict[int, str] = {}
    for hero_data in raw.values():
        hid  = hero_data.get("id")
        name = hero_data.get("localized_name", "")
        if hid and name:
            mapping[hid] = name.lower()

    log.info("Маппинг hero_id -> name загружен: %d героев.", len(mapping))
    return mapping


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
            if m.get("match_id") and m.get("start_time") and m.get("players"):
                matches.append(m)
        except (json.JSONDecodeError, TypeError):
            log.warning("Не удалось распарсить JSON одного из матчей, пропускаем.")

    log.info("Загружено %d корректных матчей из базы.", len(matches))
    return matches


# ---------------------------------------------------------------------------
# Шаг 2: Накопительный расчёт предматчевых признаков
# ---------------------------------------------------------------------------

def compute_features(
    matches: list[dict],
    hero_id_map: dict[int, str],
    hero_tags: dict[str, set[str]],
    all_tags: list[str],
) -> list[dict]:
    """
    Основная функция. Обходит матчи в хронологическом порядке и для каждого
    вычисляет предматчевые признаки на основе ПРОШЛОЙ истории.

    Ключевой принцип anti-leakage:
        1. Вычисляем признаки текущего матча из накопленной истории (прошлое).
        2. Добавляем результат текущего матча в историю (настоящее).
        3. Переходим к следующему матчу.

    Теги драфта (radiant_{tag}_count / dire_{tag}_count) — это статические
    свойства пикнутых героев, известные сразу после драфта. Data Leakage
    отсутствует: теги не зависят от исхода матча.
    """

    matches_sorted = sorted(matches, key=lambda m: m["start_time"])
    log.info("Матчи отсортированы по start_time. Начинаем вычисление признаков...")

    # История команд: team_id -> список булевых (True = победа)
    team_history: defaultdict[int, list[bool]] = defaultdict(list)

    # Статистика игрока на герое: (account_id, hero_id) -> [wins, games]
    player_hero_stats: defaultdict[tuple, list] = defaultdict(lambda: [0, 0])

    # Глобальный винрейт героя по всему датасету: hero_id -> [wins, games]
    # Используется для meta_score — насколько герой "метовый" в нашей выборке.
    # Обновляется ПОСЛЕ вычисления признаков, чтобы не было data leakage.
    hero_global_stats: defaultdict[int, list] = defaultdict(lambda: [0, 0])

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

        # -- Разделяем игроков на команды по player_slot --
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
            if not account_id or not hero_id:
                return DEFAULT_HERO_WINRATE
            wins, games = player_hero_stats[(account_id, hero_id)]
            if games == 0:
                return DEFAULT_HERO_WINRATE
            return round(wins / games, 4)

        hero_winrates: dict[str, float] = {}
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
        # Признак C: First Pick (кто сделал первый пик)
        # -------------------------------------------------------------------
        #
        # picks_bans — список действий драфта из OpenDota:
        #   [{"is_pick": bool, "team": 0|1, "order": int, ...}, ...]
        # team=0 → Radiant, team=1 → Dire.
        # Берём первое действие с is_pick=True, смотрим на team.
        # Если данных нет или первый пик определить нельзя — дефолт 0.

        def _first_pick(match_data: dict) -> int:
            picks_bans = match_data.get("picks_bans")
            if not picks_bans or not isinstance(picks_bans, list):
                return 0
            # Сортируем по order чтобы не полагаться на порядок в JSON
            picks_only = sorted(
                [a for a in picks_bans if a.get("is_pick") is True],
                key=lambda a: a.get("order", 999),
            )
            if not picks_only:
                return 0
            return 1 if picks_only[0].get("team") == 0 else 0

        radiant_first_pick = _first_pick(match)

        # -------------------------------------------------------------------
        # Признак D: Макро-теги драфта (сумма тегов по 5 героям каждой команды)
        #
        # Теги — статические свойства героев, определяются до начала матча.
        # Никакого data leakage: счётчики не зависят от исхода игры.
        # -------------------------------------------------------------------

        def _draft_tag_counts(team_players: list[dict], prefix: str) -> dict[str, int]:
            """
            Для списка игроков команды собирает hero_id, резолвит имена героев
            через hero_id_map и суммирует теги из hero_tags.
            Возвращает dict вида {prefix_tag_count: int} для всех known тегов.
            """
            counts: dict[str, int] = {f"{prefix}_{tag}_count": 0 for tag in all_tags}
            for p in team_players[:5]:
                hid       = p.get("hero_id")
                hero_name = hero_id_map.get(hid, "")
                tags      = hero_tags.get(hero_name, set())
                for tag in tags:
                    col = f"{prefix}_{tag}_count"
                    if col in counts:
                        counts[col] += 1
            return counts

        radiant_tag_counts = _draft_tag_counts(radiant_players, "radiant")
        dire_tag_counts    = _draft_tag_counts(dire_players,    "dire")

        # -------------------------------------------------------------------
        # Признак E: Synergy Score (комбо-связки в драфте)
        #
        # Проверяем 3 базовых синергии Доты. Для каждой пары тегов:
        #   +1 если хотя бы один герой команды несёт тег A
        #     И хотя бы один герой команды несёт тег B.
        # Итого: 0..3 для каждой команды.
        # Нет data leakage: теги — статические свойства героев.
        # -------------------------------------------------------------------

        SYNERGY_PAIRS = [
            ("initiation",       "burst_damage"),       # ультимативное открытие + добивание
            ("hard_save",        "late_game_scaling"),  # защита кери на поздней стадии
            ("waveclear",        "elusive"),            # безопасный сплитпуш
        ]

        def _synergy_score(team_players: list[dict]) -> int:
            """Возвращает число выполненных синерго-пар (0..3) для команды."""
            team_tags: set[str] = set()
            for p in team_players[:5]:
                hid       = p.get("hero_id")
                hero_name = hero_id_map.get(hid, "")
                team_tags |= hero_tags.get(hero_name, set())

            score = 0
            for tag_a, tag_b in SYNERGY_PAIRS:
                if tag_a in team_tags and tag_b in team_tags:
                    score += 1
            return score

        radiant_synergy_score = _synergy_score(radiant_players)
        dire_synergy_score    = _synergy_score(dire_players)

        # -------------------------------------------------------------------
        # Признак F: Meta Score (средний глобальный винрейт героев команды)
        #
        # hero_global_stats[hero_id] = [wins, games] — накопленная статистика
        # по всем матчам СТРОГО ДО текущего (обновляется после вычисления).
        # Смысл: герой, который часто побеждал в прошлых матчах нашего датасета,
        # условно "в мете" — это даёт слабый, но честный сигнал.
        # Дефолт 0.5 если герой ни разу не встречался ранее.
        # -------------------------------------------------------------------

        DEFAULT_META = 0.5

        def _meta_score(team_players: list[dict]) -> float:
            """Средний глобальный винрейт 5 пикнутых героев по прошлым матчам."""
            rates = []
            for p in team_players[:5]:
                hid = p.get("hero_id")
                if not hid:
                    rates.append(DEFAULT_META)
                    continue
                wins, games = hero_global_stats[hid]
                rates.append(round(wins / games, 4) if games > 0 else DEFAULT_META)
            # Если команда неполная — добиваем дефолтом
            while len(rates) < 5:
                rates.append(DEFAULT_META)
            return round(sum(rates) / len(rates), 4)

        radiant_meta_score = _meta_score(radiant_players)
        dire_meta_score    = _meta_score(dire_players)

        # -------------------------------------------------------------------
        # Формируем итоговую строку (только предматчевые данные!)
        # -------------------------------------------------------------------
        # radiant_winrate_advantage: положительное значение = Radiant в лучшей форме.
        # None если хотя бы одна из команд ещё не имеет истории (нет данных).
        if radiant_recent_winrate is not None and dire_recent_winrate is not None:
            winrate_advantage = round(radiant_recent_winrate - dire_recent_winrate, 4)
        else:
            winrate_advantage = None

        row = {
            "match_id":                   match_id,
            "start_time":                 start_time,
            "patch_version":              patch_version,
            "radiant_win":                radiant_win,
            "radiant_first_pick":         radiant_first_pick,
            "radiant_recent_winrate":     radiant_recent_winrate,
            "dire_recent_winrate":        dire_recent_winrate,
            "radiant_winrate_advantage":  winrate_advantage,
            "radiant_synergy_score":      radiant_synergy_score,
            "dire_synergy_score":         dire_synergy_score,
            "radiant_meta_score":         radiant_meta_score,
            "dire_meta_score":            dire_meta_score,
            **hero_winrates,
            **radiant_tag_counts,
            **dire_tag_counts,
        }
        rows_out.append(row)

        # -------------------------------------------------------------------
        # Обновляем историю ПОСЛЕ вычисления признаков — критически важно!
        # -------------------------------------------------------------------

        if radiant_team_id:
            team_history[radiant_team_id].append(radiant_win == 1)
        if dire_team_id:
            team_history[dire_team_id].append(radiant_win == 0)

        for p in players:
            acc_id  = p.get("account_id")
            hero_id = p.get("hero_id")
            if not hero_id:
                continue
            is_radiant = p.get("player_slot", 0) < 128
            won = (radiant_win == 1 and is_radiant) or (radiant_win == 0 and not is_radiant)

            # Обновляем глобальную статистику героя (для meta_score)
            g = hero_global_stats[hero_id]
            g[0] += int(won)
            g[1] += 1

            # Обновляем статистику игрока на герое (для hero_winrate)
            if acc_id:
                stats = player_hero_stats[(acc_id, hero_id)]
                stats[0] += int(won)
                stats[1] += 1

    log.info("Вычисление завершено: %d строк готово.", len(rows_out))
    return rows_out


# ---------------------------------------------------------------------------
# Шаг 3: Сохранение датасета в CSV
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict], out_path: Path) -> None:
    """
    Преобразует список строк в DataFrame и сохраняет в CSV.
    None-значения сохраняются как пустые ячейки.
    """
    if not rows:
        log.warning("Нет данных для сохранения.")
        return

    df = pd.DataFrame(rows)

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

    if not HERO_TAGS_PATH.exists():
        log.error("hero_tags_full.json не найден: %s", HERO_TAGS_PATH)
        return

    # Загружаем теги (читаем файл, НЕ модифицируем)
    hero_tags, all_tags = load_hero_tags(HERO_TAGS_PATH)

    # Загружаем маппинг hero_id -> hero_name из API
    hero_id_map = fetch_hero_id_map()
    if not hero_id_map:
        log.warning(
            "Маппинг hero_id не загружен — теги драфта будут нулевыми. "
            "Проверьте интернет-соединение."
        )

    matches = load_matches(DB_PATH)
    if not matches:
        log.error("В базе нет матчей для обработки.")
        return

    rows = compute_features(matches, hero_id_map, hero_tags, all_tags)
    save_csv(rows, OUT_CSV)
    log.info("Готово. Выходной файл: %s", OUT_CSV)


if __name__ == "__main__":
    main()
