import json
import logging
import sqlite3
import time
from pathlib import Path
import requests

# Настройка логов
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "dota_data.db"
API_BASE = "https://api.opendota.com/api"
RECENT_WINDOW = 15       # Сколько последних игр брать
RATE_LIMIT_SLEEP = 2.0   # Пауза между игроками
RETRY_SLEEP = 60         # Сколько ждать при лимите

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_table(conn):
    # 1. Создаем таблицу, если её вообще нет
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_stats_pro (
            account_id       INTEGER PRIMARY KEY,
            name             TEXT,
            last_match_id    INTEGER,
            pro_hero_json    TEXT,
            recent_pro_results TEXT,
            total_pro_games  INTEGER
        );
    """)
    # 2. Если таблица старая (осталась от прошлого запуска), добавляем колонку name
    try:
        conn.execute("ALTER TABLE player_stats_pro ADD COLUMN name TEXT")
    except sqlite3.OperationalError:
        pass # Если колонка уже есть, ошибка просто игнорируется
        
    conn.commit()

def get_pro_names():
    log.info("Загружаю справочник имен про-игроков...")
    try:
        resp = requests.get(f"{API_BASE}/proPlayers")
        if resp.status_code == 200:
            return {p['account_id']: (p.get('name') or p.get('personaname')) for p in resp.json()}
    except:
        log.error("Не удалось загрузить имена, будут использованы ID")
    return {}

def fetch_pro_matches(account_id):
    # date=365 берет матчи только за последний год
    url = f"{API_BASE}/players/{account_id}/matches?limit=500&date=365"
    while True:
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 429:
                log.warning("Лимит API! Сплю 60 секунд...")
                time.sleep(RETRY_SLEEP)
                continue
            
            data = resp.json()
            if isinstance(data, dict) and "error" in data:
                log.warning("Ошибка API, пробую подождать...")
                time.sleep(RETRY_SLEEP)
                continue
                
            if not isinstance(data, list):
                return []

            # Фильтруем только турнирные матчи
            return [m for m in data if isinstance(m, dict) and (m.get('leagueid', 0) > 0 or m.get('lobby_type') == 1)]
        except Exception as e:
            log.error(f"Ошибка запроса для {account_id}: {e}")
            return None

def build_player_data(account_id, matches, names_dict):
    matches = sorted(matches, key=lambda x: x.get("start_time", 0), reverse=True)
    total_games = len(matches)
    last_match_id = matches[0].get("match_id") if total_games > 0 else None
    
    hero_stats = {}
    recent_results = []
    
    for i, m in enumerate(matches):
        hero_id = m.get("hero_id")
        is_radiant = m.get("player_slot", 0) < 128
        win = 1 if m.get("radiant_win") == is_radiant else 0
        
        if i < RECENT_WINDOW:
            recent_results.append("W" if win else "L")
            
        if hero_id:
            if hero_id not in hero_stats:
                hero_stats[hero_id] = {"games": 0, "wins": 0}
            hero_stats[hero_id]["games"] += 1
            hero_stats[hero_id]["wins"] += win

    for hid in hero_stats:
        s = hero_stats[hid]
        s["winrate"] = round(s["wins"] / s["games"], 3)

    return {
        "account_id": account_id,
        "name": names_dict.get(account_id, f"ID:{account_id}"),
        "last_match_id": last_match_id,
        "pro_hero_json": json.dumps(hero_stats),
        "recent_pro_results": json.dumps(recent_results),
        "total_pro_games": total_games
    }

def run():
    if not DB_PATH.exists():
        log.error("База данных dota_data.db не найдена!")
        return

    conn = get_conn()
    init_table(conn)
    
    # Собираем ID игроков из твоих матчей
    rows = conn.execute("SELECT raw_data FROM match_details").fetchall()
    u_ids = set()
    for (raw,) in rows:
        d = json.loads(raw)
        for p in d.get("players", []):
            if p.get("account_id"): u_ids.add(p["account_id"])
    
    names_dict = get_pro_names()
    ids = list(u_ids)
    log.info(f"Найдено {len(ids)} уникальных игроков. Начинаю сбор (только за последний год)...")

    for i, acc_id in enumerate(ids, 1):
        matches = fetch_pro_matches(acc_id)
        time.sleep(RATE_LIMIT_SLEEP)
        
        if not matches:
            log.info(f"[{i}/{len(ids)}] У игрока с ID {acc_id} нет про-матчей за год. Пропуск.")
            continue
            
        data = build_player_data(acc_id, matches, names_dict)
        conn.execute("""
            INSERT OR REPLACE INTO player_stats_pro 
            (account_id, name, last_match_id, pro_hero_json, recent_pro_results, total_pro_games)
            VALUES (:account_id, :name, :last_match_id, :pro_hero_json, :recent_pro_results, :total_pro_games)
        """, data)
        conn.commit()
        
        log.info(f"[{i}/{len(ids)}] {data['name']} : {data['total_pro_games']} матчей за год | Форма: {' '.join(json.loads(data['recent_pro_results']))}")

    log.info("Сбор данных завершен успешно!")
    conn.close()

if __name__ == "__main__":
    run()