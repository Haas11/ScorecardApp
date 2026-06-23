"""
Grid detection for KNBSB scorecard scans.

Known scorecard structure:
 - Header row (inning numbers): half a cell tall
 - Player rows: 10 (9 batters + sub counts as same slot)
 - Inning columns: 11 (innings 1-9 filled, 10 empty, 11 has per-player stats)
 - Cells are squares; slight perspective compression handled by linear interpolation

Usage:
    python scorecard/probe_grid.py <image> [n_player_rows=10] [n_inning_cols=11]
"""
import sys
from pathlib import Path
import cv2
import numpy as np


def raw_lines_from_projection(proj: np.ndarray, threshold_frac: float = 0.15) -> list[int]:
    thresh = proj.max() * threshold_frac
    positions, in_run, start = [], False, 0
    for i, v in enumerate(proj):
        if v > thresh and not in_run:
            in_run, start = True, i
        elif v <= thresh and in_run:
            in_run = False
            positions.append((start + i) // 2)
    if in_run:
        positions.append((start + len(proj)) // 2)
    return positions


def cluster(positions: list[int], gap: int) -> list[int]:
    if not positions:
        return []
    result, group = [], [positions[0]]
    for p in positions[1:]:
        if p - group[-1] <= gap:
            group.append(p)
        else:
            result.append(sum(group) // len(group))
            group = [p]
    result.append(sum(group) // len(group))
    return result


def find_row_tops(raw_h: list[int], cell_size: int, n_rows: int) -> list[int]:
    """
    raw_h[0] = card top border
    raw_h[1] = player 1 top (bottom of the half-height inning-number header)
    Each subsequent player top is the first detected line >= prev_top + 0.7*cell_size.
    Synthesizes a boundary when no detected line is found.
    """
    if len(raw_h) < 2:
        return []
    min_step = cell_size * 0.7
    tops = [raw_h[1]]
    for _ in range(1, n_rows):
        min_next = tops[-1] + min_step
        candidates = [y for y in raw_h if y >= min_next]
        tops.append(candidates[0] if candidates else tops[-1] + cell_size)
    return tops


def detect_grid(
    img_path: str,
    n_player_rows: int = 10,
    n_inning_cols: int = 11,
    n_extra_rows: int = 2,
    debug_out: str | None = None,
    cells_out: str | None = None,
):
    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"ERROR: cannot read {img_path}")

    h, w = img.shape[:2]
    print(f"Image: {w}x{h}px  players={n_player_rows}  innings={n_inning_cols}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Cluster gap scales with resolution (0.5% of shorter dimension)
    clust_gap = max(4, min(w, h) // 200)

    # ── Horizontal lines (span >= half image width) ───────────────────────────
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 2, 1))
    h_proj = np.sum(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hk), axis=1)
    raw_h = cluster(raw_lines_from_projection(h_proj), clust_gap)

    # Cell size = median of large gaps (> 50px); excludes half-cell midlines
    gaps_h = [raw_h[i+1] - raw_h[i] for i in range(len(raw_h)-1)]
    large_gaps = [g for g in gaps_h if g > 50]
    cell_size = int(np.median(large_gaps)) if large_gaps else 80
    print(f"Cell size from rows: {cell_size}px")

    row_tops = find_row_tops(raw_h, cell_size, n_player_rows)
    row_bottoms = [t + cell_size for t in row_tops]
    print(f"Row tops:    {row_tops}")
    print(f"Row bottoms: {row_bottoms}")
    if len(row_tops) > 1:
        actual_gaps = [row_tops[i+1]-row_tops[i] for i in range(len(row_tops)-1)]
        print(f"Row gaps:    {actual_gaps}")

    # ── Vertical lines (span >= 1/3 image height) ─────────────────────────────
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 3))
    v_proj = np.sum(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vk), axis=0)
    raw_v = cluster(raw_lines_from_projection(v_proj), clust_gap)
    print(f"\nRaw vertical lines ({len(raw_v)}): {raw_v}")

    # Linear interpolation from first to last detected line handles perspective.
    # raw_v[0] = left boundary of inning 1 (confirmed reliable by user).
    # raw_v[-1] = right boundary of last inning column.
    x_start = raw_v[0]
    x_end = raw_v[-1]
    col_lefts = [
        round(x_start + i * (x_end - x_start) / n_inning_cols)
        for i in range(n_inning_cols + 1)
    ]
    col_width = (x_end - x_start) / n_inning_cols
    print(f"Inning cols: x={x_start}..{x_end}  width={col_width:.1f}px")
    print(f"Col lefts:   {col_lefts}")

    # Extra data rows (totals) below the player rows
    extra_tops = [row_bottoms[-1] + i * cell_size for i in range(n_extra_rows)]
    extra_bottoms = [t + cell_size for t in extra_tops]
    print(f"Extra row tops:  {extra_tops}")

    # ── Debug image ───────────────────────────────────────────────────────────
    if debug_out:
        dbg = img.copy()
        for i, top in enumerate(row_tops):
            cv2.line(dbg, (0, top), (w, top), (0, 0, 255), 2)
            cv2.putText(dbg, f"P{i+1}", (5, top + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        cv2.line(dbg, (0, row_bottoms[-1]), (w, row_bottoms[-1]), (0, 0, 255), 2)
        for i, top in enumerate(extra_tops):
            cv2.line(dbg, (0, top), (w, top), (0, 128, 255), 2)
            cv2.putText(dbg, f"E{i+1}", (5, top + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 200), 1)
        for ci, x in enumerate(col_lefts):
            cv2.line(dbg, (x, 0), (x, h), (255, 0, 0), 2)
            cv2.putText(dbg, str(ci + 1), (x + 2, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
        cv2.imwrite(debug_out, dbg)
        print(f"\nDebug image -> {debug_out}")

    # ── Sample cell strip: player 1, innings 1-9 ─────────────────────────────
    if cells_out and row_tops and len(col_lefts) > 9:
        crops = []
        for ci in range(9):
            y1, y2 = row_tops[0], row_bottoms[0]
            x1, x2 = col_lefts[ci], col_lefts[ci + 1]
            crops.append(img[max(0,y1):y2, max(0,x1):x2])
        if crops:
            target_h = crops[0].shape[0]
            resized = [cv2.resize(c, (target_h, target_h)) for c in crops]
            cv2.imwrite(cells_out, np.hstack(resized))
            print(f"Cell strip P1 innings 1-9 -> {cells_out}")

    return row_tops, row_bottoms, extra_tops, extra_bottoms, col_lefts, cell_size


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "images/scans/2026-06-07_almere.jpg"
    n_players = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_innings = int(sys.argv[3]) if len(sys.argv) > 3 else 11
    stem = Path(img_path).stem
    scans_dir = Path(img_path).parent
    debug_dir = scans_dir / "Gridded"
    debug_dir.mkdir(exist_ok=True)
    detect_grid(
        img_path, n_players, n_innings,
        debug_out=str(debug_dir / f"{stem}_grid_debug.png"),
        cells_out=str(debug_dir / f"{stem}_cells_row1.png"),
    )
