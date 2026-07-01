from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import yaml
from rapidfuzz import fuzz

from models import GameExtraction, PlateAppearance, PitchingLine

_CONFIG_PATH = Path(__file__).parent / "config.yml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get_data_root() -> Path:
    cfg = _load_config()
    rel = cfg.get("paths", {}).get("data_root", "../Quick 2026 data")
    return (Path(__file__).parent / rel).resolve()


# Resolved once at import time; all scripts share this path regardless of CWD.
_DB_PATH: Path = _get_data_root() / "season.db"


def _load_thresholds() -> int:
    return _load_config().get("fuzzy", {}).get("auto_match_threshold", 70)


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[.\-]", "", name)
    return name


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS players (
            player_id       INTEGER PRIMARY KEY,
            name            TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            jersey_number   INTEGER
        );

        CREATE TABLE IF NOT EXISTS player_aliases (
            alias            TEXT NOT NULL,
            normalized_alias TEXT NOT NULL,
            player_id        INTEGER REFERENCES players(player_id)
        );

        CREATE TABLE IF NOT EXISTS games (
            game_id       INTEGER PRIMARY KEY,
            date          TEXT,
            opponent      TEXT,
            game_number   TEXT,
            raw_json_path TEXT,
            imported_at   TEXT,
            UNIQUE(date, opponent, game_number)
        );

        CREATE TABLE IF NOT EXISTS plate_appearances (
            pa_id         INTEGER PRIMARY KEY,
            player_id     INTEGER REFERENCES players(player_id),
            game_id       INTEGER REFERENCES games(game_id),
            inning        INTEGER,
            batting_order INTEGER,
            result        TEXT,
            run_scored    INTEGER,
            rbi           INTEGER,
            sb            INTEGER,
            cs            INTEGER,
            bb            INTEGER,
            hp            INTEGER,
            sac           INTEGER,
            sf            INTEGER,
            raw_notes     TEXT,
            confidence    TEXT,
            needs_review  INTEGER DEFAULT 0,
            reviewed      INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS pitching_appearances (
            pa_id           INTEGER PRIMARY KEY,
            player_id       INTEGER REFERENCES players(player_id),
            game_id         INTEGER REFERENCES games(game_id),
            innings_pitched REAL,
            runs_allowed    INTEGER,
            earned_runs     INTEGER,
            strikeouts      INTEGER,
            walks           INTEGER,
            confidence      TEXT,
            needs_review    INTEGER DEFAULT 0,
            reviewed        INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS pending_aliases (
            id           INTEGER PRIMARY KEY,
            incoming     TEXT NOT NULL,
            stored       TEXT NOT NULL,
            player_id    INTEGER REFERENCES players(player_id),
            score        REAL,
            match_type   TEXT
        );
    """)
    conn.commit()
    conn.close()


def _read_roster() -> list[tuple[str, str | None]]:
    """Return [(name, jersey_str_or_None), ...] from players.txt."""
    players_txt = _get_data_root() / "players.txt"
    roster: list[tuple[str, str | None]] = []
    if not players_txt.exists():
        return roster
    for line in players_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",", 1)
        name = parts[0].strip()
        jersey = parts[1].strip() if len(parts) > 1 else None
        if name:
            roster.append((name, jersey))
    return roster


def _interactive_resolve(
    conn: sqlite3.Connection,
    detected_name: str,
    best_player,
    best_score: int,
) -> int:
    """
    Prompt the user to pick the correct player from players.txt, or create a
    new one.  Falls back to silent new-player creation when stdin is not a TTY
    (non-interactive pipelines).
    """
    # Score so low that the name bears no resemblance to any existing player
    if best_score < 30:
        norm = normalize_name(detected_name)
        cur = conn.execute(
            "INSERT INTO players (name, normalized_name, jersey_number) VALUES (?,?,?)",
            (detected_name, norm, None),
        )
        print(f"  Auto-created new player '{detected_name}' (no close match in DB)")
        return cur.lastrowid

    if not sys.stdin.isatty():
        print(
            f"WARNING: cannot interactively resolve '{detected_name}' "
            f"(non-interactive). Creating as new player.",
            file=sys.stderr,
        )
        norm = normalize_name(detected_name)
        cur = conn.execute(
            "INSERT INTO players (name, normalized_name, jersey_number) VALUES (?,?,?)",
            (detected_name, norm, None),
        )
        return cur.lastrowid

    roster = _read_roster()
    players_txt = _get_data_root() / "players.txt"

    print()
    print(f"  Unrecognized name from scorecard: '{detected_name}'")
    if best_player:
        print(f"  Best fuzzy match: '{best_player['name']}' (score: {best_score}/100)")
    print()
    print("  Select the correct player:")
    for i, (pname, jersey) in enumerate(roster, 1):
        tag = f"  #{jersey}" if jersey else ""
        print(f"    {i:>2}.  {pname}{tag}")
    new_idx = len(roster) + 1
    print(f"    {new_idx:>2}.  Enter new player name")
    print()

    while True:
        try:
            raw = input("  Choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(roster):
                chosen_name, chosen_jersey = roster[idx - 1]
                # look up (or create) in DB
                norm_chosen = normalize_name(chosen_name)
                row = conn.execute(
                    "SELECT player_id FROM players WHERE normalized_name=?",
                    (norm_chosen,),
                ).fetchone()
                if row:
                    player_id = row["player_id"]
                else:
                    jersey_int = (
                        int(chosen_jersey)
                        if chosen_jersey and chosen_jersey.isdigit()
                        else None
                    )
                    cur = conn.execute(
                        "INSERT INTO players (name, normalized_name, jersey_number) VALUES (?,?,?)",
                        (chosen_name, norm_chosen, jersey_int),
                    )
                    player_id = cur.lastrowid
                conn.execute(
                    "INSERT OR IGNORE INTO pending_aliases "
                    "(incoming, stored, player_id, score, match_type) VALUES (?,?,?,?,?)",
                    (detected_name, chosen_name, player_id, best_score, "manual"),
                )
                print(f"  Mapped '{detected_name}' → '{chosen_name}'")
                return player_id

            if idx == new_idx:
                break  # fall through to new-player entry

        print(f"  Enter a number between 1 and {new_idx}.")

    # ── New player ────────────────────────────────────────────────────────────
    while True:
        try:
            new_name = input("  New player name: ").strip()
        except (EOFError, KeyboardInterrupt):
            new_name = detected_name
            break
        if new_name:
            break
        print("  Name cannot be empty.")

    jersey_raw = ""
    try:
        jersey_raw = input("  Jersey number (leave blank to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        pass
    jersey_int = int(jersey_raw) if jersey_raw.isdigit() else None

    norm_new = normalize_name(new_name)
    cur = conn.execute(
        "INSERT INTO players (name, normalized_name, jersey_number) VALUES (?,?,?)",
        (new_name, norm_new, jersey_int),
    )
    player_id = cur.lastrowid

    # Append to players.txt
    jersey_suffix = f", {jersey_int}" if jersey_int is not None else ""
    with open(players_txt, "a", encoding="utf-8") as f:
        f.write(f"{new_name}{jersey_suffix}\n")

    print(f"  Created '{new_name}' and added to {players_txt.name}")
    return player_id


def _resolve_player(
    conn: sqlite3.Connection,
    name: str,
    jersey_number: Optional[int],
    auto_thresh: int,
) -> int:
    norm = normalize_name(name)

    # 1. Exact match on normalized_name
    row = conn.execute(
        "SELECT player_id FROM players WHERE normalized_name = ?", (norm,)
    ).fetchone()
    if row:
        return row["player_id"]

    # 2. Exact match on aliases
    row = conn.execute(
        "SELECT player_id FROM player_aliases WHERE normalized_alias = ?", (norm,)
    ).fetchone()
    if row:
        return row["player_id"]

    # 3. Fuzzy match
    all_players = conn.execute(
        "SELECT player_id, name, normalized_name FROM players"
    ).fetchall()
    best_score = 0
    best_player = None
    for p in all_players:
        score = fuzz.token_sort_ratio(norm, p["normalized_name"])
        if score > best_score:
            best_score = score
            best_player = p

    if best_player and best_score >= auto_thresh:
        print(f"Auto-matched '{name}' → '{best_player['name']}' ({best_score})")
        conn.execute(
            "INSERT INTO pending_aliases (incoming, stored, player_id, score, match_type) VALUES (?,?,?,?,?)",
            (name, best_player["name"], best_player["player_id"], best_score, "auto"),
        )
        return best_player["player_id"]

    # Below auto-threshold: ask the user instead of silently guessing
    return _interactive_resolve(conn, name, best_player, best_score)


def find_duplicate_game(
    conn: sqlite3.Connection,
    date: Optional[str],
    opponent: Optional[str],
    game_number: Optional[str],
) -> Optional[int]:
    if game_number is None:
        row = conn.execute(
            "SELECT game_id FROM games WHERE date=? AND opponent=? AND game_number IS NULL",
            (date, opponent),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT game_id FROM games WHERE date=? AND opponent=? AND game_number=?",
            (date, opponent, game_number),
        ).fetchone()
    return row["game_id"] if row else None


def mark_reviewed(conn: sqlite3.Connection, game_id: int) -> int:
    """Set reviewed=1, needs_review=0 for all PAs of a game. Returns row count."""
    n = conn.execute(
        "UPDATE plate_appearances SET reviewed=1, needs_review=0 WHERE game_id=?",
        (game_id,),
    ).rowcount
    conn.commit()
    return n


def delete_game(conn: sqlite3.Connection, game_id: int) -> None:
    conn.execute("DELETE FROM plate_appearances WHERE game_id=?", (game_id,))
    conn.execute("DELETE FROM pitching_appearances WHERE game_id=?", (game_id,))
    conn.execute("DELETE FROM games WHERE game_id=?", (game_id,))
    conn.commit()


def write_game(
    conn: sqlite3.Connection,
    extraction: GameExtraction,
    raw_json_path: str,
    opponent_override: Optional[str] = None,
    date_override: Optional[str] = None,
) -> int:
    from datetime import datetime

    auto_thresh = _load_thresholds()

    game = extraction.game
    date_val = date_override or game.date
    teams = game.teams
    opponent_val = opponent_override or teams.get("away") or teams.get("home")

    cur = conn.execute(
        """INSERT INTO games (date, opponent, game_number, raw_json_path, imported_at)
           VALUES (?,?,?,?,?)""",
        (
            date_val,
            opponent_val,
            game.game_number,
            raw_json_path,
            datetime.utcnow().isoformat(),
        ),
    )
    game_id = cur.lastrowid

    # Clear pending aliases before import
    conn.execute("DELETE FROM pending_aliases")

    for slot in extraction.lineup:
        batting_order = slot.batting_order
        for player_entry in slot.players:
            player_id = _resolve_player(
                conn,
                player_entry.name,
                player_entry.jersey_number,
                auto_thresh,
            )
            for pa in player_entry.plate_appearances:
                result = pa.result.upper().strip()
                needs_review = 1 if pa.confidence == "low" else 0
                bb = 1 if result == "BB" else 0
                hp = 1 if result == "HP" else 0
                sac = 1 if result in ("SAC", "SH") else 0
                sf = 1 if result == "SF" else 0
                sb = pa.sb
                cs = pa.cs

                conn.execute(
                    """INSERT INTO plate_appearances
                       (player_id, game_id, inning, batting_order, result,
                        run_scored, rbi, sb, cs, bb, hp, sac, sf,
                        raw_notes, confidence, needs_review)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        player_id, game_id, pa.inning, batting_order, result,
                        int(pa.run_scored), pa.rbi, sb, cs, bb, hp, sac, sf,
                        pa.notes, pa.confidence, needs_review,
                    ),
                )

    for pitch in extraction.pitching:
        pitcher_id = _resolve_player(
            conn, pitch.name, None, auto_thresh
        )
        needs_review = 1 if pitch.confidence == "low" else 0
        conn.execute(
            """INSERT INTO pitching_appearances
               (player_id, game_id, innings_pitched, runs_allowed, earned_runs,
                strikeouts, walks, confidence, needs_review)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                pitcher_id, game_id,
                pitch.innings_pitched, pitch.runs_allowed, pitch.earned_runs,
                pitch.strikeouts, pitch.walks, pitch.confidence, needs_review,
            ),
        )

    conn.commit()
    return game_id
