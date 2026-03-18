import sqlite3
import requests
import json
import time
import logging
from pathlib import Path

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "dota_data.db"
API_BASE = "https://api.opendota.com/api"

LEAGUE_WHITELIST_KEYWORDS = [
    "dreamleague", "esl one", "the international",
    "riyadh masters", "esports world cup", "major",
    "wallachia", "pgl",
]
QUALIFIER_KEYWORDS = ["qualifier", "oq", "cq", " qual"]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS leagues (
            league_id INTEGER PRIMARY KEY,
            name      TEXT,
            tier      TEXT
        );
        CREATE TABLE IF NOT EXISTS matches (
            match_id   INTEGER PRIMARY KEY,
            league_id  INTEGER,
            start_time INTEGER
        );
        CREATE TABLE IF NOT EXISTS match_details (
            match_id INTEGER PRIMARY KEY,
            raw_data TEXT,
            patch    TEXT
        );
    """)
    conn.commit()
    log.info("Database ready: %s", DB_PATH)


# ---------------------------------------------------------------------------
# Step 1 — League whitelist
# ---------------------------------------------------------------------------

def _is_approved(league: dict) -> bool:
    tier = (league.get("tier") or "").lower()
    name = (league.get("name") or "").lower()

    if tier == "excluded":
        return False
    if any(kw in name for kw in QUALIFIER_KEYWORDS):
        return False
    if tier == "premium":
        return True
    if any(kw in name for kw in LEAGUE_WHITELIST_KEYWORDS):
        return True
    return False


def update_leagues(conn: sqlite3.Connection) -> list[int]:
    log.info("Fetching league list from OpenDota...")
    resp = requests.get(f"{API_BASE}/leagues", timeout=30)
    resp.raise_for_status()
    leagues = resp.json()

    approved = [lg for lg in leagues if _is_approved(lg)]
    log.info("Approved leagues: %d / %d total", len(approved), len(leagues))

    conn.executemany(
        "INSERT OR REPLACE INTO leagues (league_id, name, tier) VALUES (?, ?, ?)",
        [(lg["leagueid"], lg.get("name"), lg.get("tier")) for lg in approved],
    )
    conn.commit()

    ids = [lg["leagueid"] for lg in approved]
    log.info("League whitelist updated (%d leagues).", len(ids))
    return ids


# ---------------------------------------------------------------------------
# Step 2 — Collect match IDs via Explorer
# ---------------------------------------------------------------------------

def collect_match_ids(conn: sqlite3.Connection, league_ids: list[int]) -> int:
    if not league_ids:
        log.warning("No league IDs — skipping match collection.")
        return 0

    chunk_size = 20
    chunks = [league_ids[i:i + chunk_size] for i in range(0, len(league_ids), chunk_size)]
    total_chunks = len(chunks)
    cutoff_time = int(time.time()) - (90 * 24 * 60 * 60)
    log.info(
        "Collecting match IDs in %d batches (chunk size %d, since cutoff %d)...",
        total_chunks, chunk_size, cutoff_time,
    )

    inserted = 0
    for batch_num, chunk in enumerate(chunks, start=1):
        ids_str = ",".join(map(str, chunk))
        sql = (
            f"SELECT match_id, leagueid as league_id, start_time "
            f"FROM matches WHERE leagueid IN ({ids_str}) "
            f"AND start_time >= {cutoff_time} "
            f"ORDER BY start_time DESC"
        )

        log.info("Fetching batch %d/%d (%d leagues)...", batch_num, total_chunks, len(chunk))
        resp = requests.get(f"{API_BASE}/explorer", params={"sql": sql}, timeout=60)
        resp.raise_for_status()
        rows = resp.json().get("rows", [])
        log.info("Batch %d/%d returned %d matches.", batch_num, total_chunks, len(rows))

        for row in rows:
            cur = conn.execute(
                "INSERT OR IGNORE INTO matches (match_id, league_id, start_time) VALUES (?, ?, ?)",
                (row["match_id"], row["league_id"], row.get("start_time")),
            )
            inserted += cur.rowcount
        conn.commit()

        time.sleep(1.2)

    log.info("New matches added to DB: %d", inserted)
    return inserted


# ---------------------------------------------------------------------------
# Step 3 — Fetch detailed match data
# ---------------------------------------------------------------------------

def _fetch_match(match_id: int) -> dict | None:
    """Fetch a single match, retrying once on rate-limit (429)."""
    url = f"{API_BASE}/matches/{match_id}"
    while True:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            log.warning("Rate limited (429) on match %d — sleeping 60s...", match_id)
            time.sleep(60)
            continue  # retry same match
        log.error("Unexpected status %d for match %d — skipping.", resp.status_code, match_id)
        return None


def fetch_match_details(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT m.match_id FROM matches m
        LEFT JOIN match_details md ON m.match_id = md.match_id
        WHERE md.match_id IS NULL
        ORDER BY m.start_time DESC
        """
    ).fetchall()

    total = len(rows)
    if total == 0:
        log.info("No new matches to fetch details for.")
        return

    log.info("Fetching details for %d matches...", total)

    for idx, (match_id,) in enumerate(rows, start=1):
        log.info("[%d/%d] Fetching match %d", idx, total, match_id)

        data = _fetch_match(match_id)
        if data is None:
            continue

        patch = str(data.get("patch", ""))
        raw = json.dumps(data, ensure_ascii=False)

        conn.execute(
            "INSERT OR REPLACE INTO match_details (match_id, raw_data, patch) VALUES (?, ?, ?)",
            (match_id, raw, patch),
        )
        conn.commit()

        time.sleep(1.2)  # stay well under 60 req/min

    log.info("Done fetching match details.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    with get_conn() as conn:
        init_db(conn)

        # Step 1
        league_ids = update_leagues(conn)

        # Step 2
        collect_match_ids(conn, league_ids)

        # Step 3
        fetch_match_details(conn)

    log.info("All done. Database: %s", DB_PATH)


if __name__ == "__main__":
    main()
