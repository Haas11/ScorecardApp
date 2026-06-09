from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sqlite3
import yaml

_CONFIG_PATH = Path("config.yml")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


@dataclass
class PlayerStats:
    player_id: int
    name: str
    small_sample: bool

    # Counting
    G: int = 0
    PA: int = 0
    AB: int = 0
    H: int = 0
    doubles: int = 0
    triples: int = 0
    HR: int = 0
    R: int = 0
    RBI: int = 0
    BB: int = 0
    K: int = 0
    SB: int = 0
    CS: int = 0
    HP: int = 0
    SAC: int = 0
    SF: int = 0

    # Rate
    AVG: float = 0.0
    OBP: float = 0.0
    SLG: float = 0.0
    OPS: float = 0.0

    # Advanced
    BABIP: Optional[float] = None
    ISO: float = 0.0
    BB_pct: float = 0.0
    K_pct: float = 0.0
    wOBA: Optional[float] = None
    RC: float = 0.0
    OPS_plus: float = 0.0
    AB_per_HR: Optional[float] = None
    BB_K: Optional[float] = None


def compute_player_stats(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
) -> PlayerStats:
    cfg = _load_config()
    weights = cfg["league"]["woba_weights"]
    lg_obp = cfg["league"]["avg_obp"]
    lg_slg = cfg["league"]["avg_slg"]
    small_thresh = cfg["stats"]["small_sample_threshold"]

    rows = conn.execute(
        """SELECT pa.result, pa.run_scored, pa.rbi, pa.bb, pa.hp,
                  pa.sac, pa.sf, pa.sb, pa.cs, pa.game_id
           FROM plate_appearances pa
           WHERE pa.player_id = ?""",
        (player_id,),
    ).fetchall()

    G = len(
        {r["game_id"] for r in rows}
    )
    PA = len(rows)
    BB = sum(r["bb"] for r in rows)
    HP = sum(r["hp"] for r in rows)
    SAC = sum(r["sac"] for r in rows)
    SF = sum(r["sf"] for r in rows)
    R = sum(r["run_scored"] for r in rows)
    RBI = sum(r["rbi"] for r in rows)
    SB = sum(r["sb"] for r in rows)
    CS = sum(r["cs"] for r in rows)

    singles = sum(1 for r in rows if r["result"] == "1B")
    doubles = sum(1 for r in rows if r["result"] == "2B")
    triples = sum(1 for r in rows if r["result"] == "3B")
    HR = sum(1 for r in rows if r["result"] == "HR")
    H = singles + doubles + triples + HR
    K = sum(1 for r in rows if r["result"] in ("K", "KL"))
    AB = PA - BB - HP - SAC - SF

    AVG = H / AB if AB > 0 else 0.0
    obp_denom = AB + BB + HP + SF
    OBP = (H + BB + HP) / obp_denom if obp_denom > 0 else 0.0
    TB = singles + 2 * doubles + 3 * triples + 4 * HR
    SLG = TB / AB if AB > 0 else 0.0
    OPS = OBP + SLG

    babip_denom = AB - K - HR + SF
    BABIP = (H - HR) / babip_denom if babip_denom > 0 else None

    ISO = SLG - AVG
    BB_pct = BB / PA if PA > 0 else 0.0
    K_pct = K / PA if PA > 0 else 0.0

    woba_denom = AB + BB + SF + HP
    if woba_denom > 0:
        wOBA = (
            weights["bb"] * BB
            + weights["hp"] * HP
            + weights["single"] * singles
            + weights["double"] * doubles
            + weights["triple"] * triples
            + weights["hr"] * HR
        ) / woba_denom
    else:
        wOBA = None

    rc_denom = AB + BB
    RC = (H + BB) * TB / rc_denom if rc_denom > 0 else 0.0

    lg_slg_val = lg_slg if lg_slg else 0.001
    OPS_plus = 100 * (OBP / lg_obp + SLG / lg_slg_val - 1) if lg_obp > 0 else 0.0

    AB_per_HR = AB / HR if HR > 0 else None
    BB_K = BB / K if K > 0 else None

    return PlayerStats(
        player_id=player_id,
        name=player_name,
        small_sample=PA < small_thresh,
        G=G, PA=PA, AB=AB, H=H,
        doubles=doubles, triples=triples, HR=HR,
        R=R, RBI=RBI, BB=BB, K=K, SB=SB, CS=CS, HP=HP, SAC=SAC, SF=SF,
        AVG=AVG, OBP=OBP, SLG=SLG, OPS=OPS,
        BABIP=BABIP, ISO=ISO, BB_pct=BB_pct, K_pct=K_pct,
        wOBA=wOBA, RC=RC, OPS_plus=OPS_plus, AB_per_HR=AB_per_HR, BB_K=BB_K,
    )


def compute_all_stats(conn: sqlite3.Connection) -> list[PlayerStats]:
    players = conn.execute("SELECT player_id, name FROM players").fetchall()
    results = []
    for p in players:
        stats = compute_player_stats(conn, p["player_id"], p["name"])
        results.append(stats)
    return results
