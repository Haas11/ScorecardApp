"""
Grid detection for KNBSB scorecard scans.

Known scorecard structure:
 - Header row (inning numbers): half a cell tall
 - Player rows: 10 (9 batters + sub counts as same slot)
 - Inning columns: 11 (innings 1-9 filled, 10 empty, 11 has per-player stats)
 - Cells are squares; slight perspective compression handled by linear interpolation
 - Each cell has a 2×2 sub-row/sub-col structure, so horizontal lines appear
   at both cell boundaries AND at the mid-point of each cell.

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


def _snap(target: int, candidates: list[int], tol: int) -> int:
    """Return nearest candidate within tol, else return target (synthesized)."""
    if not candidates:
        return target
    nearest = min(candidates, key=lambda x: abs(x - target))
    return nearest if abs(nearest - target) <= tol else target


def _build_row_tops(raw_h: list[int], cell_size: int, n_rows: int, n_extra_rows: int = 2) -> list[int]:
    """
    Build player row top y-positions by snapping expected grid positions to
    detected horizontal lines.  cell_size is the FULL player-row height
    (already corrected for sub-row detection).

    Two card layouts:
      A) raw_h[0] is the outer top border ABOVE the header row.
         Player row 1 top = first line after the extra header line(s).
      B) raw_h[0] IS already the first player row top.

    Layout A is detected by counting detected lines: when sub-row lines are
    present, each row produces 2 lines.  If more lines are detected than
    (n_rows + n_extra_rows) * 2 expects, the surplus lines are card-border /
    header lines at the top (Layout A).  This handles both "ring bands at
    bottom" (card border near y=0) and "ring bands at top" (card border
    further down, e.g. y=126 for 1952px-tall scans).
    """
    if not raw_h:
        return [i * cell_size for i in range(n_rows)]

    grid_top = raw_h[0]
    snap_tol = cell_size // 3

    def snap(t: int) -> int:
        return _snap(t, raw_h, snap_tol)

    # Detect whether sub-row dividers are present (first gap ≈ cell_size/2).
    first_gap = (raw_h[1] - raw_h[0]) if len(raw_h) > 1 else cell_size
    has_sub_rows = first_gap < cell_size * 0.75

    if has_sub_rows:
        # Each row (player + extra) produces 2 detected lines.
        # Any surplus beyond that count must be card-border / header lines at
        # the top → Layout A.  Cap at 1: at most one extra header line expected.
        expected = (n_rows + n_extra_rows) * 2
        n_offset = 1 if len(raw_h) > expected else 0
    else:
        # No sub-rows: one line per row boundary.
        expected = n_rows + n_extra_rows
        n_offset = 1 if len(raw_h) > expected else 0

    if n_offset:
        # Layout A: raw_h[n_offset] is the first player row top.
        first_top = snap(raw_h[n_offset])
        print(f"Layout A: card border at y={raw_h[0]}; first player row at y={raw_h[n_offset]}")
    elif grid_top < cell_size * 0.4:
        # Legacy fallback: card border very close to image top, sub-rows may
        # not have pushed the count above the threshold.
        first_top = snap(grid_top + cell_size // 2)
        print(f"Layout A (legacy): first player row at y={first_top}")
    else:
        first_top = grid_top
        print(f"Layout B: first player row at y={first_top}")

    tops = [first_top]
    for _ in range(1, n_rows):
        tops.append(snap(tops[-1] + cell_size))
    return tops


# Legacy alias kept for any external callers
def find_row_tops(raw_h: list[int], cell_size: int, n_rows: int) -> list[int]:
    return _build_row_tops(raw_h, cell_size, n_rows)


def detect_grid(
    img_path: str,
    n_player_rows: int = 10,
    n_inning_cols: int = 11,   # kept for backward compat but no longer used
    n_extra_rows: int = 2,
    left_skip_frac: float = 0.05,
    debug_out: str | None = None,
    cells_out: str | None = None,
):
    """
    Detect the batting-grid layout of a KNBSB scorecard scan.

    left_skip_frac — fraction of image width to skip before V-line detection
                     (skips ring-binder area).  Default 0.05 is safe for all
                     standard KNBSB scans; x_start is auto-detected from the
                     V-line spacing pattern, so no manual tuning is needed.

    Returns (row_tops, row_bottoms, extra_tops, extra_bottoms, col_lefts, cell_size).
    All coordinates are pixel integers into the original image.
    col_lefts has len = n_detected_scoring_cols + 1.
    """
    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"ERROR: cannot read {img_path}")

    h, w = img.shape[:2]
    print(f"Image: {w}x{h}px  players={n_player_rows}  innings={n_inning_cols}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    clust_gap = max(4, min(w, h) // 200)

    # ── Horizontal lines (span >= half image width) ───────────────────────────
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 2, 1))
    h_proj = np.sum(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hk), axis=1)
    raw_h = cluster(raw_lines_from_projection(h_proj), clust_gap)

    # ── Determine true player-row height ─────────────────────────────────────
    # KNBSB cells have a 2×2 sub-structure, so morphological detection often
    # finds lines at both major row boundaries AND at the sub-row mid-points.
    # If the median detected gap is "sub-row scale" (< image_height / (rows*2+2)),
    # then the real player-row height is 2× that gap.
    gaps_h = [raw_h[i + 1] - raw_h[i] for i in range(len(raw_h) - 1)] if len(raw_h) > 1 else []
    if gaps_h:
        median_gap = int(np.median(gaps_h))
        sub_row_threshold = h // (n_player_rows * 2 + 2)
        if median_gap < sub_row_threshold:
            # Sub-row dividers are being picked up; full row = 2 × sub-row gap
            cell_size = median_gap * 2
            print(f"Sub-row lines detected (gap={median_gap}px); cell_size={cell_size}px")
        else:
            large = [g for g in gaps_h if g > h // (n_player_rows * 4)]
            cell_size = int(np.median(large)) if large else h // (n_player_rows + 2)
            print(f"Full-row lines detected; cell_size={cell_size}px")
    else:
        cell_size = h // (n_player_rows + 2)
        print(f"No H-lines detected; cell_size fallback={cell_size}px")

    print(f"H-lines ({len(raw_h)}): {raw_h[:8]}{'...' if len(raw_h) > 8 else ''}")

    row_tops = _build_row_tops(raw_h, cell_size, n_player_rows, n_extra_rows)
    row_bottoms = [t + cell_size for t in row_tops]
    print(f"Row tops:    {row_tops}")
    if len(row_tops) > 1:
        actual_gaps = [row_tops[i + 1] - row_tops[i] for i in range(len(row_tops) - 1)]
        print(f"Row gaps:    {actual_gaps}")

    # ── Vertical lines (skip player-info area on the left) ────────────────────
    # The left portion of each scorecard contains player position, name and
    # jersey-number columns before the inning columns begin.  We skip everything
    # to the left of left_skip_frac × image_width.
    left_skip = int(w * left_skip_frac)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 3))
    v_proj = np.sum(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vk), axis=0)
    raw_v_all = cluster(raw_lines_from_projection(v_proj), clust_gap)
    raw_v = [x for x in raw_v_all if x >= left_skip]
    if len(raw_v) > 1:
        gaps_v = [raw_v[i + 1] - raw_v[i] for i in range(len(raw_v) - 1)]
        print(f"V-lines total={len(raw_v_all)}, after left_skip({left_skip}px)={len(raw_v)}, "
              f"V-gaps min={min(gaps_v)} median={int(np.median(gaps_v))} max={max(gaps_v)}")
    else:
        gaps_v = []
        print(f"V-lines total={len(raw_v_all)}, after left_skip({left_skip}px)={len(raw_v)}")

    if len(raw_v) >= 3 and gaps_v:
        # Sub-column gap = modal (median) spacing between adjacent V-lines.
        # All KNBSB cards have a 2×2 sub-cell structure, so each inning column
        # produces 2 V-lines: the column-left border and a center sub-divider.
        modal_gap = int(np.median(gaps_v))
        col_width = 2 * modal_gap

        # Auto-detect x_start (left edge of inning 1) by finding the start of
        # the longest run of evenly-spaced V-lines.  The player-info area to
        # the left has irregular spacing; the inning grid has consistent spacing
        # at ≈ modal_gap.  The longest such run is the inning area.
        tol = modal_gap * 0.35
        regular = [abs(g - modal_gap) <= tol for g in gaps_v]
        best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
        for i, r in enumerate(regular):
            if r:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_len = 0
        start_pos = best_start

        # The longest run may be extended to the left by player-info dividers
        # whose sub-column gap (≈modal_gap) accidentally falls within the wide
        # tolerance.  Those false-start lines produce a two-step (col_width) gap
        # that is visibly off, while true inning boundaries are within ≈4% of
        # col_width.  Advance start_pos until the two-step gap is tight.
        tight_tol = col_width * 0.04          # ≈5.7px for 142px columns
        sp = best_start
        while sp + 2 < len(raw_v):
            two_step = raw_v[sp + 2] - raw_v[sp]
            if abs(two_step - col_width) <= tight_tol:
                break
            sp += 1
        start_pos = sp
        x_start = raw_v[start_pos]

        # Build col_lefts: take every-other V-line from x_start onward.
        # This gives all detected scoring-column left boundaries.
        col_lefts = list(raw_v[start_pos::2])
        if len(col_lefts) < 2:
            col_lefts.append(col_lefts[-1] + col_width)

        x_end = col_lefts[-1]
        n_detected = len(col_lefts) - 1
        print(f"Inning cols: x={x_start}..{x_end}  col_width={col_width}px  ({n_detected} cols detected)")
    else:
        # Fallback: proportional estimate (1 col ≈ w/16)
        col_w_est = w // 16
        x_end = w - w // 20
        x_start = max(left_skip, x_end - n_inning_cols * col_w_est)
        col_lefts = [
            round(x_start + i * (x_end - x_start) / n_inning_cols)
            for i in range(n_inning_cols + 1)
        ]
        print(f"V-lines fallback: x_start={x_start}  x_end={x_end}")

    print(f"Col lefts:   {col_lefts}")

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

    # ── Sample cell strip: player 1, innings 1–min(9, n_inning_cols) ──────────
    if cells_out and row_tops and len(col_lefts) > 1:
        crops = []
        for ci in range(min(9, n_inning_cols)):
            y1, y2 = row_tops[0], row_bottoms[0]
            x1, x2 = col_lefts[ci], col_lefts[ci + 1]
            crops.append(img[max(0, y1):y2, max(0, x1):x2])
        if crops:
            target_h = crops[0].shape[0]
            resized = [cv2.resize(c, (target_h, target_h)) for c in crops]
            cv2.imwrite(cells_out, np.hstack(resized))
            print(f"Cell strip P1 innings 1-{len(crops)} -> {cells_out}")

    return row_tops, row_bottoms, extra_tops, extra_bottoms, col_lefts, cell_size


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "images/scans/2026-06-07_almere.jpg"
    n_players = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_innings = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    stem = Path(img_path).stem
    scans_dir = Path(img_path).parent
    debug_dir = scans_dir / "Gridded"
    debug_dir.mkdir(exist_ok=True)
    detect_grid(
        img_path, n_players, n_innings,
        debug_out=str(debug_dir / f"{stem}_grid_debug.png"),
        cells_out=str(debug_dir / f"{stem}_cells_row1.png"),
    )
