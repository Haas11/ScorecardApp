#!/usr/bin/env python3
"""
render_widget.py — Generate a self-contained HTML scorecard widget from extraction JSON.

Usage (standalone):
    uv run python scorecard/render_widget.py images/data/raw/STEM_cells.json
    # → writes images/data/widgets/STEM.html and opens it in the browser

Called automatically by extract_cells.py at the end of each run.
"""
from __future__ import annotations

import base64
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
            h   = summary.get("H")  if summary.get("H")  is not None else sum(1 for pa in pas if _is_hit(pa.get("result")))
            ab  = summary.get("AB") if summary.get("AB") is not None else sum(1 for pa in pas if _is_ab(pa.get("result")))
            r   = summary.get("R")  if summary.get("R")  is not None else sum(1 for pa in pas if pa.get("run_scored"))
            bb  = sum(1 for pa in pas if (pa.get("result") or "").upper() == "BB")
            hbp = sum(1 for pa in pas if (pa.get("result") or "").upper() in ("HP", "HBP"))
            sf  = sum(1 for pa in pas if (pa.get("result") or "").upper() == "SF")
            tb  = sum(
                {"1B": 1, "2B": 2, "3B": 3, "HR": 4}.get((pa.get("result") or "").upper(), 0)
                for pa in pas
            )
            obp_denom = ab + bb + hbp + sf
            slg_denom = ab
            avg_str = f".{round(h  / ab         * 1000):03d}" if ab         > 0 else ".---"
            obp_str = f".{round((h + bb + hbp) / obp_denom * 1000):03d}" if obp_denom > 0 else ".---"
            slg_str = f".{round(tb / slg_denom  * 1000):03d}" if slg_denom > 0 else ".---"
            entry_inning = (
                min((pa.get("inning", 0) for pa in pas), default=None)
                if idx > 0 else 1
            )
            players.append({
                "name":          player.get("name", "?"),
                "num":           player.get("jersey_number"),
                "PA":            summary.get("PA", pa_count),
                "AB":            ab,
                "H":             h,
                "R":             r,
                "AVG":           avg_str,
                "OBP":           obp_str,
                "SLG":           slg_str,
                "batting_order": slot.get("batting_order", 0),
                "is_sub":        idx > 0,
                "entry_inning":  entry_inning,
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
    total_pa  = sum(p["PA"] for p in players)
    total_r   = sum(p["R"]  for p in players)
    total_h   = sum(p["H"]  for p in players)
    total_ab  = sum(p["AB"] for p in players)
    # Re-derive BB/HBP/SF/TB from raw PA lists for accurate team totals
    _all_pas = [pa for slot in lineup for pl in slot.get("players", []) for pa in pl.get("plate_appearances", [])]
    total_bb  = sum(1 for pa in _all_pas if (pa.get("result") or "").upper() == "BB")
    total_hbp = sum(1 for pa in _all_pas if (pa.get("result") or "").upper() in ("HP", "HBP"))
    total_sf  = sum(1 for pa in _all_pas if (pa.get("result") or "").upper() == "SF")
    total_tb  = sum({"1B":1,"2B":2,"3B":3,"HR":4}.get((pa.get("result") or "").upper(), 0) for pa in _all_pas)
    _obp_d    = total_ab + total_bb + total_hbp + total_sf
    avg = f".{round(total_h / total_ab * 1000):03d}" if total_ab > 0 else ".000"
    obp = f".{round((total_h + total_bb + total_hbp) / _obp_d * 1000):03d}" if _obp_d > 0 else ".000"
    slg = f".{round(total_tb / total_ab * 1000):03d}" if total_ab > 0 else ".000"

    game_info = game.get("game", {})
    teams     = game_info.get("teams", {})
    home      = teams.get("home", "Home")
    away      = teams.get("away", "Away")
    date      = game_info.get("date") or ""
    title     = f"{home} vs {away}" + (f" — {date}" if date else "")

    # last_batter_by_inning: inning (str key) → 1-based batting slot of last batter
    last_batter = {int(k): v for k, v in game.get("last_batter_by_inning", {}).items()}

    return {
        "title":   title,
        "home":    home,
        "away":    away,
        "date":    date,
        "players": players,
        "innings": inning_data,
        "totals":  {"PA": total_pa, "R": total_r, "H": total_h, "AB": total_ab, "AVG": avg, "OBP": obp, "SLG": slg},
        "last_batter_by_inning": last_batter,
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
  --bg:  #ffffff; --bg2: #f8fafc; --bg3: #f1f5f9; --bg-bench: #d8dfe8;
  --tx:  #0f172a; --tx2: #64748b; --tx3: #94a3b8;
  --brd: rgba(15,23,42,0.12);
  --rad: 8px;
  --hdr: rgb(31,95,160);
  --s-bg: #dcfce7; --s-tx: #15803d;
  --i-bg: #dbeafe; --i-tx: #1d4ed8;
  --d-bg: #fee2e2; --d-tx: #b91c1c;
  --w-bg: #fef9c3; --w-tx: #a16207;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px; color: var(--tx); background: var(--bg);
}
@media (prefers-color-scheme: dark) { :root {
  --bg:  #0f172a; --bg2: #1e293b; --bg3: #334155; --bg-bench: #475569;
  --tx:  #f1f5f9; --tx2: #94a3b8; --tx3: #64748b;
  --brd: rgba(241,245,249,0.12);
  --hdr: rgb(31,95,160);
  --s-bg: #14532d; --s-tx: #86efac;
  --i-bg: #1e3a5f; --i-tx: #93c5fd;
  --d-bg: #450a0a; --d-tx: #fca5a5;
  --w-bg: #422006; --w-tx: #fcd34d;
} }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { padding: 1.25rem; max-width: 960px; margin: 0 auto; }
h1 { font-size: 15px; font-weight: 500; color: var(--tx2); margin-bottom: 1.25rem; }
.cards { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 10px; margin-bottom: 1.25rem; }
.card { background: var(--hdr); border-radius: var(--rad); padding: 0.75rem 1rem; color: #fff; }
.card-lbl { font-size: 11px; color: rgba(255,255,255,0.75); margin-bottom: 3px; }
.card-val { font-size: 22px; font-weight: 500; }
.card-sub { font-size: 10px; color: rgba(255,255,255,0.65); }
.banner { background: var(--w-bg); color: var(--w-tx); border-radius: var(--rad);
  padding: 7px 12px; margin-bottom: 1rem; font-size: 12px; display: flex; align-items: center; gap: 8px; }
table { width: 100%; border-collapse: collapse; table-layout: fixed; }
th, td { border: 0.5px solid var(--brd); padding: 3px; text-align: center; vertical-align: middle; }
th { font-size: 11px; font-weight: 500; color: var(--tx2); background: var(--bg2); padding: 4px 3px; }
thead th { background: var(--hdr); color: #fff; }
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
.bench{ background: var(--bg-bench); }
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
<div id="dbgimg" style="margin-top:1.5rem"></div>
<p id="bugnote" style="display:none" class="note">
  &#x26A0; One or more cells show <s>null</s> — VLM returned string "null" instead of JSON null.
</p>
<div class="legend">
  <span><span class="sw" style="background:var(--s-bg)"></span>Hit</span>
  <span><span class="sw" style="background:var(--i-bg)"></span>BB / HP</span>
  <span><span class="sw" style="background:var(--d-bg)"></span>Out</span>
  <span><span class="sw" style="background:var(--w-bg)"></span>FC / E / SF</span>
  <span><span class="sw" style="background:var(--bg2);border:0.5px solid var(--brd)"></span>No PA</span>
  <span><span class="sw" style="background:var(--bg-bench);border:0.5px solid var(--brd)"></span>Inactive (sub not entered / starter exited)</span>
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

function paCell(pas, isSub, entryInning, inning, extraStyle, exitInning) {
  const s = extraStyle ? ` style="${extraStyle}"` : '';
  if (!pas || pas.length === 0) {
    if (isSub && entryInning != null && inning < entryInning)
      return `<td class="bench"${s}></td>`;
    if (!isSub && exitInning != null && inning >= exitInning)
      return `<td class="bench"${s}></td>`;
    return `<td class="mpt"${s}><span class="etx">&#x2014;</span></td>`;
  }
  if (pas.length === 1) {
    const p = pas[0], ct = ctype(p.r);
    const dot = p.run ? ' <span class="run">&#x25CF;</span>' : '';
    const txt = p.r === 'null' ? '<s>null</s>' : (p.r || '—');
    return `<td class="${BGCLS[ct]}"${s}><span class="${TXCLS[ct]}">${txt}${dot}</span></td>`;
  }
  const items = pas.map(p => {
    const ct = ctype(p.r);
    const dot = p.run ? ' <span class="run">&#x25CF;</span>' : '';
    const txt = p.r === 'null' ? '<s>null</s>' : (p.r || '—');
    return `<div class="blk ${BGCLS[ct]}"><span class="${TXCLS[ct]}">${txt}${dot}</span></div>`;
  }).join('');
  const baseStyle = extraStyle ? `padding:2px;${extraStyle}` : 'padding:2px';
  return `<td style="${baseStyle}"><div style="display:flex;flex-direction:column">${items}</div></td>`;
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
let hdr = '<tr><th style="width:20px;font-size:10px;padding:2px">#</th><th class="pl-th">Player</th>';
D.innings.forEach(s => {
  const badge = s.wrap ? `<div class="wbdg">wrap</div>` : '';
  hdr += `<th style="width:44px">${s.inn}${badge}</th>`;
});
const stH = 'style="width:30px;font-size:11px;text-align:center"';
hdr += `<th ${stH}>PA</th><th ${stH}>H</th><th ${stH}>AB</th><th ${stH}>R</th>`;
hdr += `<th ${stH}>AVG</th><th ${stH}>OBP</th><th ${stH}>SLG</th></tr>`;
document.getElementById('shead').innerHTML = hdr;

// Body
// last_batter_by_inning: {inning -> 1-based batting slot of last batter in that inning}
// lastPlayerInSlot: batting_order -> highest D.players index for that slot (last sub, or starter)
const lastBatterByInning = D.last_batter_by_inning || {};
const lastPlayerInSlot = {};
D.players.forEach((p, pi) => { lastPlayerInSlot[p.batting_order] = pi; });

// subEntryInning: batting_order -> inning the first sub enters (for greying out starter cells)
const subEntryInning = {};
D.players.forEach(p => {
  if (p.is_sub && p.entry_inning != null) {
    if (subEntryInning[p.batting_order] == null || p.entry_inning < subEntryInning[p.batting_order])
      subEntryInning[p.batting_order] = p.entry_inning;
  }
});

let hasBug = false, body = '';
D.players.forEach((p, pi) => {
  const tag = p.is_sub
    ? `<span class="sub">&#x21B3;</span>${p.name}`
    : `<span class="num">#${p.num ?? '?'}</span>${p.name}`;
  const slotCell = p.is_sub
    ? `<td style="padding:2px"></td>`
    : `<td style="text-align:center;font-size:11px;font-weight:500;color:var(--tx3);padding:2px">${p.batting_order}</td>`;
  const slotBorder = (!p.is_sub && p.batting_order > 1) ? 'border-top:2px solid var(--brd)' : '';
  const exitInning = (!p.is_sub) ? (subEntryInning[p.batting_order] ?? null) : null;
  let row = `<tr style="${slotBorder}">${slotCell}<td class="pl-td">${tag}</td>`;
  inns.forEach(i => {
    const pas = p.cells[String(i)] || null;
    // Bold bottom border marks the last batter of each inning
    const isLastBatter = lastBatterByInning[i] === p.batting_order && lastPlayerInSlot[p.batting_order] === pi;
    const botBorder = isLastBatter ? 'border-bottom:2px solid var(--tx2);' : '';
    const slotTop   = (!p.is_sub && p.batting_order > 1) ? 'border-top:2px solid var(--brd);' : '';
    const cellStyle = slotTop + botBorder;
    row += paCell(pas, p.is_sub, p.entry_inning, i, cellStyle, exitInning);
    if (pas && pas.some(x => x.r === 'null')) hasBug = true;
  });
  const rc = p.R > 0 ? 'rok' : 'rzr';
  const sb = (!p.is_sub && p.batting_order > 1) ? 'border-top:2px solid var(--brd);' : '';
  const stc = `${sb}width:30px;font-size:11px;text-align:center;padding:2px 3px`;
  row += `<td style="${stc};color:var(--tx2)">${p.PA}</td>`;
  row += `<td style="${stc};color:var(--tx2)">${p.H}</td>`;
  row += `<td style="${stc};color:var(--tx2)">${p.AB}</td>`;
  row += `<td style="${stc}" class="${rc}">${p.R}</td>`;
  row += `<td style="${stc};color:var(--tx2)">${p.AVG}</td>`;
  row += `<td style="${stc};color:var(--tx2)">${p.OBP}</td>`;
  row += `<td style="${stc};color:var(--tx2)">${p.SLG}</td></tr>`;
  body += row;
});
document.getElementById('sbody').innerHTML = body;

// Footer
let foot = '<tr style="border-top:2px solid var(--brd)"><td style="padding:2px"></td><td class="pl-td" style="font-weight:500;color:var(--tx2)">Totals</td>';
D.innings.forEach(s => {
  const rc = s.R > 0 ? 'rok' : 'rzr';
  foot += `<td class="fcel"><div>PA${s.PA}</div><div class="${rc}">R${s.R}</div><div>H${s.H}</div></td>`;
});
const t = D.totals;
const ftd = 'width:30px;font-size:11px;text-align:center;padding:2px 3px;font-weight:500';
foot += `<td style="${ftd};color:var(--tx2)">${t.PA}</td>`;
foot += `<td style="${ftd};color:var(--tx2)">${t.H}</td>`;
foot += `<td style="${ftd};color:var(--tx2)">${t.AB}</td>`;
foot += `<td style="${ftd}" class="rok">${t.R}</td>`;
foot += `<td style="${ftd};color:var(--tx2)">${t.AVG}</td>`;
foot += `<td style="${ftd};color:var(--tx2)">${t.OBP}</td>`;
foot += `<td style="${ftd};color:var(--tx2)">${t.SLG}</td></tr>`;
document.getElementById('sfoot').innerHTML = foot;

if (hasBug) document.getElementById('bugnote').style.display = 'block';

if (D.debug_img_b64) {
  document.getElementById('dbgimg').innerHTML =
    '<p style="font-size:11px;color:var(--tx3);margin-bottom:6px">Grid detection debug</p>' +
    `<img src="data:image/png;base64,${D.debug_img_b64}" style="max-width:100%;border-radius:var(--rad);border:0.5px solid var(--brd)">`;
}
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


def render_widget_for_game(game: dict, out_path: Path, debug_img_path: Path | None = None) -> Path:
    """
    Compute stats from a parsed _cells.json dict and write the HTML widget.
    Returns the output path.
    """
    stats = compute_stats(game)
    stats["debug_img_b64"] = None
    if debug_img_path and debug_img_path.exists():
        stats["debug_img_b64"] = base64.b64encode(debug_img_path.read_bytes()).decode()
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
    debug_img = json_path.parent / f"{stem}_grid_debug.png"

    render_widget_for_game(game, out_path, debug_img_path=debug_img)
    print(f"Widget: {out_path}")

    if "--no-open" not in sys.argv:
        webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
