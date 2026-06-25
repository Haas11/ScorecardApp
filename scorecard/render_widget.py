#!/usr/bin/env python3
"""
render_widget.py — Generate a self-contained HTML scorecard widget from extraction JSON.

Usage (standalone):
    uv run python scorecard/render_widget.py images/data/raw/STEM_cells.json
    # → writes images/data/widgets/STEM.html and opens it in the browser

Called automatically by extract_cells.py at the end of each run.
"""
from __future__ import annotations

import json
import re
import sys
import webbrowser
from pathlib import Path

_HITS   = {"1B", "2B", "3B", "HR"}
_NOT_AB = {"BB", "HP", "HBP", "SAC", "SH", "SF"}


# ── Stat helpers ──────────────────────────────────────────────────────────────

def _cell_type(r: str | None) -> str:
    if not r:
        return "empty"
    if r == "null":
        return "bug"
    u = r.upper()
    if u in _HITS:
        return "hit"
    if u in ("BB", "HP", "HBP"):
        return "reach"
    if re.match(r"^E\d", r, re.IGNORECASE) or u in ("FC", "K-PB"):
        return "warn"
    return "out"


def _is_hit(r: str | None) -> bool:
    return (r or "").upper() in _HITS


def _is_ab(r: str | None) -> bool:
    return r is not None and (r or "").upper() not in _NOT_AB


# ── Data computation ──────────────────────────────────────────────────────────

def compute_stats(game: dict) -> dict:
    """
    Derive all display data from a parsed _cells.json dict.
    Returns a JSON-serialisable dict ready to embed in the HTML template.
    """
    lineup = game.get("lineup", [])

    # ── Per-player data ───────────────────────────────────────────────────────
    players = []
    for slot in lineup:
        slot_players = slot.get("players", [])
        for idx, player in enumerate(slot_players):
            pas = player.get("plate_appearances", [])
            cells_by_inning: dict[str, list[dict]] = {}
            for pa in pas:
                key = str(pa.get("inning", 0))
                cells_by_inning.setdefault(key, []).append({
                    "r":   pa.get("result"),
                    "run": bool(pa.get("run_scored")),
                })
            summary = player.get("summary") or {}
            pa_count = len(pas)
            h  = summary.get("H")  if summary.get("H")  is not None else sum(1 for pa in pas if _is_hit(pa.get("result")))
            ab = summary.get("AB") if summary.get("AB") is not None else sum(1 for pa in pas if _is_ab(pa.get("result")))
            r  = summary.get("R")  if summary.get("R")  is not None else sum(1 for pa in pas if pa.get("run_scored"))
            players.append({
                "name":          player.get("name", "?"),
                "num":           player.get("jersey_number"),
                "PA":            summary.get("PA", pa_count),
                "AB":            ab,
                "H":             h,
                "R":             r,
                "batting_order": slot.get("batting_order", 0),
                "is_sub":        idx > 0,
                "cells":         cells_by_inning,
            })

    # ── Per-inning aggregates ─────────────────────────────────────────────────
    inning_set: set[int] = set()
    for slot in lineup:
        for player in slot.get("players", []):
            for pa in player.get("plate_appearances", []):
                inn = pa.get("inning", 0)
                if inn > 0:
                    inning_set.add(inn)

    n_slots   = len(lineup)
    inning_data = []
    for inn in sorted(inning_set):
        pa_count = h_count = r_count = 0
        for slot in lineup:
            for player in slot.get("players", []):
                for pa in player.get("plate_appearances", []):
                    if pa.get("inning") != inn:
                        continue
                    if pa.get("result") is not None:
                        pa_count += 1
                    if pa.get("run_scored"):
                        r_count += 1
                    if _is_hit(pa.get("result")):
                        h_count += 1
        inning_data.append({
            "inn":  inn,
            "PA":   pa_count,
            "R":    r_count,
            "H":    h_count,
            "wrap": pa_count > n_slots,
        })

    # ── Totals ────────────────────────────────────────────────────────────────
    total_pa = sum(p["PA"] for p in players)
    total_r  = sum(p["R"]  for p in players)
    total_h  = sum(p["H"]  for p in players)
    total_ab = sum(p["AB"] for p in players)
    avg = f".{round(total_h / total_ab * 1000):03d}" if total_ab > 0 else ".000"

    game_info = game.get("game", {})
    teams     = game_info.get("teams", {})
    home      = teams.get("home", "Home")
    away      = teams.get("away", "Away")
    date      = game_info.get("date") or ""
    title     = f"{home} vs {away}" + (f" — {date}" if date else "")

    return {
        "title":   title,
        "home":    home,
        "away":    away,
        "date":    date,
        "players": players,
        "innings": inning_data,
        "totals":  {"PA": total_pa, "R": total_r, "H": total_h, "AB": total_ab, "AVG": avg},
    }


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/dist/tabler-icons.min.css">
<style>
:root {
  --bg:  #ffffff; --bg2: #f8fafc; --bg3: #f1f5f9;
  --tx:  #0f172a; --tx2: #64748b; --tx3: #94a3b8;
  --brd: rgba(15,23,42,0.12);
  --rad: 8px;
  --s-bg: #dcfce7; --s-tx: #15803d;
  --i-bg: #dbeafe; --i-tx: #1d4ed8;
  --d-bg: #fee2e2; --d-tx: #b91c1c;
  --w-bg: #fef9c3; --w-tx: #a16207;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px; color: var(--tx); background: var(--bg);
}
@media (prefers-color-scheme: dark) { :root {
  --bg:  #0f172a; --bg2: #1e293b; --bg3: #334155;
  --tx:  #f1f5f9; --tx2: #94a3b8; --tx3: #64748b;
  --brd: rgba(241,245,249,0.12);
  --s-bg: #14532d; --s-tx: #86efac;
  --i-bg: #1e3a5f; --i-tx: #93c5fd;
  --d-bg: #450a0a; --d-tx: #fca5a5;
  --w-bg: #422006; --w-tx: #fcd34d;
} }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { padding: 1.25rem; max-width: 960px; margin: 0 auto; }
h1 { font-size: 15px; font-weight: 500; color: var(--tx2); margin-bottom: 1.25rem; }
.cards { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 10px; margin-bottom: 1.25rem; }
.card { background: var(--bg2); border-radius: var(--rad); padding: 0.75rem 1rem; }
.card-lbl { font-size: 11px; color: var(--tx2); margin-bottom: 3px; }
.card-val { font-size: 22px; font-weight: 500; }
.card-sub { font-size: 10px; color: var(--tx3); }
.banner { background: var(--w-bg); color: var(--w-tx); border-radius: var(--rad);
  padding: 7px 12px; margin-bottom: 1rem; font-size: 12px; display: flex; align-items: center; gap: 8px; }
table { width: 100%; border-collapse: collapse; table-layout: fixed; }
th, td { border: 0.5px solid var(--brd); padding: 3px; text-align: center; vertical-align: middle; }
th { font-size: 11px; font-weight: 500; color: var(--tx2); background: var(--bg2); padding: 4px 3px; }
.pl-th { text-align: left; padding-left: 8px; width: 120px; }
.pl-td { text-align: left; padding: 3px 4px 3px 8px; font-size: 12px; white-space: nowrap; }
.sm-td { text-align: left; padding: 3px 6px; font-size: 11px; white-space: nowrap; width: 90px; }
.num  { font-size: 10px; color: var(--tx3); margin-right: 3px; }
.sub  { font-size: 10px; color: var(--i-tx); margin-right: 3px; }
.hit  { background: var(--s-bg); }
.out  { background: var(--d-bg); }
.rch  { background: var(--i-bg); }
.wrn  { background: var(--w-bg); }
.mpt  { background: var(--bg2); }
.bug  { background: var(--bg2); }
.htx  { color: var(--s-tx); font-size: 11px; font-weight: 500; }
.otx  { color: var(--d-tx); font-size: 11px; font-weight: 500; }
.rtx  { color: var(--i-tx); font-size: 11px; font-weight: 500; }
.wtx  { color: var(--w-tx); font-size: 11px; font-weight: 500; }
.etx  { color: var(--tx3);  font-size: 11px; }
.btx  { color: var(--tx3);  font-size: 11px; text-decoration: line-through; }
.run  { color: var(--s-tx); font-size: 8px; }
.blk  { border-radius: 2px; padding: 1px 2px; margin-bottom: 1px; }
.fcel { font-size: 10px; color: var(--tx2); line-height: 1.65; }
.wbdg { font-size: 8px; background: var(--w-bg); color: var(--w-tx);
  border-radius: 2px; padding: 0 3px; display: inline-block; }
.rok  { color: var(--s-tx); font-weight: 500; }
.rzr  { color: var(--tx3); }
.legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11px; color: var(--tx2); margin-top: 1rem; }
.sw { width: 11px; height: 11px; border-radius: 2px; display: inline-block; vertical-align: middle; margin-right: 3px; }
.note { font-size: 11px; color: var(--tx3); margin-top: 0.75rem; padding: 6px 10px;
  background: var(--bg2); border-radius: var(--rad); }
</style>
</head>
<body>
<h1 id="ttl"></h1>
<div class="cards" id="cards"></div>
<div id="wbanner" style="display:none" class="banner">
  <i class="ti ti-arrows-exchange" style="font-size:15px" aria-hidden="true"></i>
  <span id="wtext"></span>
</div>
<div style="overflow-x:auto">
<table>
  <thead id="shead"></thead>
  <tbody id="sbody"></tbody>
  <tfoot id="sfoot"></tfoot>
</table>
</div>
<p id="bugnote" style="display:none" class="note">
  &#x26A0; One or more cells show <s>null</s> — VLM returned string "null" instead of JSON null.
</p>
<div class="legend">
  <span><span class="sw" style="background:var(--s-bg)"></span>Hit</span>
  <span><span class="sw" style="background:var(--i-bg)"></span>BB / HP</span>
  <span><span class="sw" style="background:var(--d-bg)"></span>Out</span>
  <span><span class="sw" style="background:var(--w-bg)"></span>FC / E / SF</span>
  <span><span class="sw" style="background:var(--bg2);border:0.5px solid var(--brd)"></span>No PA</span>
  <span class="rok">&#x25CF; run scored</span>
</div>
<script>
const D = __DATA_JSON__;
const BGCLS  = {hit:'hit', out:'out', reach:'rch', warn:'wrn', empty:'mpt', bug:'bug'};
const TXCLS  = {hit:'htx', out:'otx', reach:'rtx', warn:'wtx', empty:'etx', bug:'btx'};

function ctype(r) {
  if (!r) return 'empty';
  if (r === 'null') return 'bug';
  const u = r.toUpperCase();
  if (['1B','2B','3B','HR'].includes(u)) return 'hit';
  if (['BB','HP','HBP'].includes(u)) return 'reach';
  if (/^E\d/i.test(r) || ['FC','K-PB'].includes(u)) return 'warn';
  return 'out';
}

function paCell(pas) {
  if (!pas || pas.length === 0)
    return '<td class="mpt"><span class="etx">&#x2014;</span></td>';
  if (pas.length === 1) {
    const p = pas[0], ct = ctype(p.r);
    const dot = p.run ? ' <span class="run">&#x25CF;</span>' : '';
    const txt = p.r === 'null' ? '<s>null</s>' : (p.r || '—');
    return `<td class="${BGCLS[ct]}"><span class="${TXCLS[ct]}">${txt}${dot}</span></td>`;
  }
  const items = pas.map(p => {
    const ct = ctype(p.r);
    const dot = p.run ? ' <span class="run">&#x25CF;</span>' : '';
    const txt = p.r === 'null' ? '<s>null</s>' : (p.r || '—');
    return `<div class="blk ${BGCLS[ct]}"><span class="${TXCLS[ct]}">${txt}${dot}</span></div>`;
  }).join('');
  return `<td style="padding:2px"><div style="display:flex;flex-direction:column">${items}</div></td>`;
}

document.getElementById('ttl').textContent = D.title;

// Metric cards
document.getElementById('cards').innerHTML = [
  ['Plate appearances', D.totals.PA, '',     ''],
  ['Runs scored',       D.totals.R,  'rok',  ''],
  ['Hits',              D.totals.H,  '',     ''],
  ['Batting avg',       D.totals.AVG,'',     D.totals.H + 'H / ' + D.totals.AB + 'AB'],
].map(([l, v, c, s]) =>
  `<div class="card"><div class="card-lbl">${l}</div>` +
  `<div class="card-val ${c}">${v}</div>` +
  (s ? `<div class="card-sub">${s}</div>` : '') + `</div>`
).join('');

// Wrap banner
const wraps = D.innings.filter(s => s.wrap);
if (wraps.length) {
  document.getElementById('wbanner').style.display = 'flex';
  document.getElementById('wtext').textContent =
    'Wrap detected: inning ' + wraps.map(s => `${s.inn} (${s.PA} PA)`).join(', ');
}

// Header
const inns = D.innings.map(s => s.inn);
let hdr = '<tr><th class="pl-th">Player</th>';
D.innings.forEach(s => {
  const badge = s.wrap ? `<div class="wbdg">wrap</div>` : '';
  hdr += `<th style="width:44px">${s.inn}${badge}</th>`;
});
hdr += '<th class="sm-td" style="text-align:left;padding-left:6px">PA&#xB7;H&#xB7;AB&#xB7;R</th></tr>';
document.getElementById('shead').innerHTML = hdr;

// Body
let hasBug = false, body = '';
D.players.forEach(p => {
  const tag = p.is_sub
    ? `<span class="sub">&#x21B3;</span>${p.name}`
    : `<span class="num">#${p.num ?? '?'}</span>${p.name}`;
  let row = `<tr><td class="pl-td">${tag}</td>`;
  inns.forEach(i => {
    const pas = p.cells[String(i)] || null;
    row += paCell(pas);
    if (pas && pas.some(x => x.r === 'null')) hasBug = true;
  });
  const rc = p.R > 0 ? 'rok' : 'rzr';
  row += `<td class="sm-td"><span style="color:var(--tx2)">${p.PA}&#xB7;${p.H}&#xB7;${p.AB}&#xB7;</span><span class="${rc}">${p.R}</span></td></tr>`;
  body += row;
});
document.getElementById('sbody').innerHTML = body;

// Footer
let foot = '<tr style="border-top:1px solid var(--brd)"><td class="pl-td" style="font-weight:500;color:var(--tx2)">Totals</td>';
D.innings.forEach(s => {
  const rc = s.R > 0 ? 'rok' : 'rzr';
  foot += `<td class="fcel"><div>PA${s.PA}</div><div class="${rc}">R${s.R}</div><div>H${s.H}</div></td>`;
});
const t = D.totals;
foot += `<td class="sm-td"><span style="color:var(--tx2)">${t.PA}&#xB7;${t.H}&#xB7;${t.AB}&#xB7;</span><span class="rok">${t.R}</span></td></tr>`;
document.getElementById('sfoot').innerHTML = foot;

if (hasBug) document.getElementById('bugnote').style.display = 'block';
</script>
</body>
</html>
"""


# ── Public API ────────────────────────────────────────────────────────────────

def render_html(stats: dict) -> str:
    """Return a complete self-contained HTML string for the given stats dict."""
    data_json = json.dumps(stats, ensure_ascii=False)
    return (
        _HTML
        .replace("__TITLE__", stats.get("title", "Scorecard"))
        .replace("__DATA_JSON__", data_json)
    )


def render_widget_for_game(game: dict, out_path: Path) -> Path:
    """
    Compute stats from a parsed _cells.json dict and write the HTML widget.
    Returns the output path.
    """
    stats = compute_stats(game)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(stats), encoding="utf-8")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: render_widget.py <path_to_cells.json> [--no-open]", file=sys.stderr)
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"ERROR: {json_path} not found", file=sys.stderr)
        sys.exit(1)

    game = json.loads(json_path.read_text(encoding="utf-8"))
    out_dir  = json_path.parent.parent / "widgets"
    stem     = json_path.stem.removesuffix("_cells")
    out_path = out_dir / f"{stem}.html"

    render_widget_for_game(game, out_path)
    print(f"Widget: {out_path}")

    if "--no-open" not in sys.argv:
        webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
