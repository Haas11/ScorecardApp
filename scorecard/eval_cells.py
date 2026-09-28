"""
VLM cell-reading eval set (#0).

Labels come from the per-cell cache (r##_c##.json) of hand-verified games; crops
use the golden grid geometry from grid_golden.json (the exact crops the
verified reads came from). Re-runs the current prompts on every cell and
reports accuracy, so a prompt/crop change can be judged instead of guessed.

Run from scorecard/:
    uv run python eval_cells.py --check        # no API: label set + cache vs _cells.json consistency
    uv run python eval_cells.py                # full eval (result/run via classify_cell, rbi via _read_rbi_slot)
    uv run python eval_cells.py --only rbi     # just the RBI reader on run cells
    uv run python eval_cells.py --save-crops   # also write mistaken crops to eval_out/ for inspection
Each run writes eval_out/<timestamp>.json with every prediction.
"""
import concurrent.futures
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import click
import cv2

HERE = Path(__file__).parent
SEASON = HERE.parent / "Quick 2026"
OUT = HERE / "eval_out"
EVAL_GAMES = [
    "2026-07-12 Grizzlies (Home)",
    "2026-08-23 Urbanus (Home)",
]
_ALIASES = {"HBP": "HP", "SH": "SAC"}


def _norm(r):
    if r is None:
        return None
    r = str(r).strip().upper().replace(" ", "")
    return _ALIASES.get(r, r)


def load_set() -> list[dict]:
    """One item per grid cell of every eval game (including empty cells)."""
    golden = json.loads((HERE / "grid_golden.json").read_text())
    items = []
    for stem in EVAL_GAMES:
        cells = SEASON / "games" / stem / "cells"
        layout = json.loads((cells / "_layout.json").read_text())
        innings = layout["run_args"]["innings"]
        col_to_inning = layout["col_to_inning"]
        names = json.loads((cells / "_names.json").read_text(encoding="utf-8"))
        g = golden[stem]
        for cf in sorted(cells.glob("r??_c??.json")):
            ri, ci = int(cf.stem[1:3]) - 1, int(cf.stem[5:7]) - 1
            if ci >= len(col_to_inning) or col_to_inning[ci] > innings or ci + 1 >= len(g["col_lefts"]):
                continue
            c = json.loads(cf.read_text(encoding="utf-8"))
            if str(c.get("notes") or "").startswith("api_error"):
                continue
            res = _norm(c.get("result"))
            items.append({
                "game": stem, "cell": cf.stem, "ri": ri, "ci": ci, "inning": col_to_inning[ci],
                "name": (names.get(f"r{ri+1:02d}", {}).get("players") or [f"P{ri+1}"])[0],
                "box": (g["row_tops"][ri], g["row_bottoms"][ri], g["col_lefts"][ci], g["col_lefts"][ci + 1]),
                "result": res,
                "run": bool(c.get("run")) if res else False,
                # HR bottom-left digit is the batter's own (or absent) -- not an RBI label.
                "rbi_slot": c.get("rbi_slot") if res and c.get("run") and res != "HR" and "rbi_slot" in c else "n/a",
            })
    return items


def check_consistency(items: list[dict]) -> int:
    """Compare cache-derived PAs with the hand-checked _cells.json (per row+inning)."""
    bad = 0
    for stem in EVAL_GAMES:
        d = json.loads((SEASON / "games" / stem / f"{stem}_cells.json").read_text(encoding="utf-8"))
        want = defaultdict(list)
        for slot in d["lineup"]:
            for p in slot["players"]:
                for pa in p["plate_appearances"]:
                    want[(slot["batting_order"], pa["inning"])].append((_norm(pa["result"]), bool(pa["run_scored"])))
        got = defaultdict(list)
        for it in items:
            if it["game"] == stem and it["result"]:
                got[(it["ri"] + 1, it["inning"])].append((it["result"], it["run"]))
        for k in sorted(set(want) | set(got)):
            if sorted(want[k]) != sorted(got[k]):
                bad += 1
                print(f"  MISMATCH {stem} slot {k[0]} inn {k[1]}: _cells.json={sorted(want[k])} cache={sorted(got[k])}")
    return bad


def _crop(item, imgs):
    y1, y2, x1, x2 = item["box"]
    return imgs[item["game"]][max(0, y1):y2, max(0, x1):x2]


@click.command()
@click.option("--check", is_flag=True, help="No API calls: print the label set and verify cache == _cells.json.")
@click.option("--only", type=click.Choice(["all", "cell", "rbi"]), default="all")
@click.option("--model", default=None, help="Default: EXTRACTION_MODEL env.")
@click.option("--workers", default=8)
@click.option("--thinking", type=int, default=None,
              help="Override classify_cell's Gemini thinking budget (and give it room: max_tokens += budget).")
@click.option("--save-crops", is_flag=True, help="Write crops of mistaken cells to eval_out/<run>/.")
def main(check, only, model, workers, save_crops, thinking):
    items = load_set()
    pa = [i for i in items if i["result"]]
    rbi = [i for i in pa if i["rbi_slot"] != "n/a"]
    print(f"Eval set: {len(items)} cells ({len(pa)} PAs, {sum(i['run'] for i in pa)} runs, "
          f"{len(rbi)} RBI-labelled run cells) from {len(EVAL_GAMES)} games")
    print(f"Results: {dict(Counter(i['result'] for i in pa).most_common())}")
    n_bad = check_consistency(items)
    print(f"Cache vs _cells.json: {'consistent' if not n_bad else f'{n_bad} mismatch(es) — fix labels first'}")
    import extract_cells as ec
    imgs = {s: cv2.imread(str(SEASON / "scans" / f"{s}.jpg")) for s in EVAL_GAMES}
    ink = {(i["game"], i["cell"]): ec._pen_ink(_crop(i, imgs), 0.08) for i in items}
    skip = [i for i in items if ink[(i["game"], i["cell"])] < ec._EMPTY_INK_MAX]
    lost = [f"{i['game'][:10]} {i['cell']} {i['result']}" for i in skip if i["result"]]
    print(f"Empty-cell skip: {len(skip)}/{len(items)} cells skipped without API, "
          f"{len(lost)} real PAs lost{': ' + ', '.join(lost) if lost else ''}")
    if check:
        return

    import extract_cells as ec  # loads .env; import late so --check needs no key
    model = model or os.environ.get("EXTRACTION_MODEL", "gemini-2.5-flash")
    if model.startswith("gemini"):
        from google import genai
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if thinking is not None:
        ec._CELL_THINKING_BUDGET = thinking
        ec._CELL_MAX_TOKENS = 400 + thinking
    print(f"classify_cell: thinking_budget={ec._CELL_THINKING_BUDGET} max_tokens={ec._CELL_MAX_TOKENS}")
    imgs = {s: cv2.imread(str(SEASON / "scans" / f"{s}.jpg")) for s in EVAL_GAMES}

    def run_cell(it):
        r = ec.classify_cell(_crop(it, imgs), it["name"], it["inning"], client, model)
        return {"result": _norm(r.get("result")), "run": bool(r.get("run")),
                "result_conf": r.get("result_conf"), "run_conf": r.get("run_conf"), "notes": r.get("notes"),
                "adjusted": r.get("adjusted") or []}

    def run_rbi(it):
        return ec._read_rbi_slot(_crop(it, imgs), it["name"], it["inning"], client, model)

    t0 = time.monotonic()
    preds = {}
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        if only in ("all", "cell"):
            for it, p in zip(items, ex.map(run_cell, items)):
                preds.setdefault((it["game"], it["cell"]), {})["cell"] = p
        if only in ("all", "rbi"):
            for it, p in zip(rbi, ex.map(run_rbi, rbi)):
                preds.setdefault((it["game"], it["cell"]), {})["rbi"] = p
    print(f"\n{model}: {len(preds)} cells in {time.monotonic() - t0:.0f}s")

    errors = []
    if only in ("all", "cell"):
        presence = sum((preds[(i["game"], i["cell"])]["cell"]["result"] is None) == (i["result"] is None) for i in items)
        res_ok = sum(preds[(i["game"], i["cell"])]["cell"]["result"] == i["result"] for i in pa)
        run_ok = sum(preds[(i["game"], i["cell"])]["cell"]["run"] == i["run"] for i in pa)
        runs = [i for i in pa if i["run"]]
        run_recall = sum(preds[(i["game"], i["cell"])]["cell"]["run"] for i in runs)
        run_fp = sum(preds[(i["game"], i["cell"])]["cell"]["run"] for i in pa if not i["run"])
        print(f"PA presence : {presence}/{len(items)} ({presence / len(items):.1%})")
        print(f"Result      : {res_ok}/{len(pa)} ({res_ok / len(pa):.1%})")
        n_conf = sum(preds[(i["game"], i["cell"])]["cell"]["result_conf"] is not None for i in items)
        flags = Counter(f.split(":")[0] for i in items for f in preds[(i["game"], i["cell"])]["cell"]["adjusted"])
        print(f"Checks fired: {dict(flags) or 'none'}")
        print(f"Self-conf   : {n_conf}/{len(items)} reads include result_conf")
        print(f"Run         : {run_ok}/{len(pa)} ({run_ok / len(pa):.1%})  "
              f"recall {run_recall}/{len(runs)}, false runs {run_fp}")
        for i in items:
            p = preds[(i["game"], i["cell"])]["cell"]
            if p["result"] != i["result"] or (i["result"] and p["run"] != i["run"]):
                errors.append((i, f"result {i['result']}/{i['run']} -> {p['result']}/{p['run']} "
                                  f"(conf {p['result_conf']}/{p['run_conf']}) {p['notes'] or ''}"))
    if only in ("all", "rbi") and rbi:
        rbi_ok = sum(preds[(i["game"], i["cell"])]["rbi"] == i["rbi_slot"] for i in rbi)
        print(f"RBI slot    : {rbi_ok}/{len(rbi)} ({rbi_ok / len(rbi):.1%})")
        for i in rbi:
            p = preds[(i["game"], i["cell"])]["rbi"]
            if p != i["rbi_slot"]:
                errors.append((i, f"rbi {i['rbi_slot']} -> {p}"))

    OUT.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for i, msg in errors:
            print(f"  {i['game'][:22]:22s} {i['cell']} inn {i['inning']}: {msg}")
    if save_crops and errors:
        d = OUT / stamp
        d.mkdir()
        for i, msg in errors:
            cv2.imwrite(str(d / f"{i['game'][:10]}_{i['cell']}.png"), _crop(i, imgs))
        print(f"Crops -> {d}")
    (OUT / f"{stamp}.json").write_text(json.dumps(
        {"model": model, "only": only,
         "preds": {f"{g}|{c}": v for (g, c), v in preds.items()},
         "errors": [f"{i['game']}|{i['cell']}: {m}" for i, m in errors]}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
