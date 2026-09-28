"""
Perspective/skew rectification for scorecard photos (#16, phase 1).

Fits the printed grid's long straight lines (adaptive threshold -> short
morphological line kernels -> connected components -> cv2.fitLine), takes the
outermost long horizontal and vertical lines, intersects them into a quad and
warps that quad to an axis-aligned rectangle. The whole image is warped (not
cropped), so the result looks like an aligned scan and probe_grid.detect_grid
runs on it unchanged.

Any two horizontal + two vertical printed lines define the same physical
rectangle family, so which exact border lines get picked does not matter for
straightening — only that they are long and straight.

Also provides fit_grid() (lattice fit of the KNBSB grid to the line mask) and
detect_grid_lattice(), the default grid detector used by extract_cells.py.

Usage:
    python rectify.py <image> [out_image]
"""
import sys

import cv2
import numpy as np

from probe_grid import cluster, raw_lines_from_projection


def _fit_lines(mask: np.ndarray, horizontal: bool, min_span: float) -> list[tuple[float, float, float]]:
    """Fit one line per long connected component.

    Returns (slope, intercept, span): y = slope*x + intercept for horizontal
    lines, x = slope*y + intercept for vertical ones.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lines = []
    for i in range(1, n):
        span = stats[i, cv2.CC_STAT_WIDTH] if horizontal else stats[i, cv2.CC_STAT_HEIGHT]
        if span < min_span:
            continue
        ys, xs = np.nonzero(labels == i)
        a, b = (xs, ys) if horizontal else (ys, xs)
        slope, intercept = np.polyfit(a.astype(np.float64), b.astype(np.float64), 1)
        lines.append((float(slope), float(intercept), float(span)))
    return lines


def _intersect(hl: tuple, vl: tuple) -> tuple[float, float]:
    """Intersect y = m1*x + c1 with x = m2*y + c2."""
    m1, c1 = hl[0], hl[1]
    m2, c2 = vl[0], vl[1]
    y = (m1 * c2 + c1) / (1 - m1 * m2)
    return m2 * y + c2, y


def find_grid_quad(img: np.ndarray) -> tuple[np.ndarray, dict] | tuple[None, dict]:
    """Return the 4 corners (TL, TR, BR, BL) of the outermost long grid lines, or None."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    block = max(15, (min(w, h) // 40) | 1)
    binv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, block, 15)
    # Short kernels survive a few degrees of tilt; a later close bridges
    # small gaps (ink crossings, faint print) so each printed line is one component.
    hk = max(20, w // 60)
    vk = max(20, h // 40)
    hmask = cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    hmask = cv2.morphologyEx(hmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 3)))
    vmask = cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    vmask = cv2.morphologyEx(vmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, vk)))

    hlines = _fit_lines(hmask, True, w * 0.4)
    vlines = _fit_lines(vmask, False, h * 0.4)
    info = {"n_h": len(hlines), "n_v": len(vlines)}
    if len(hlines) < 2 or len(vlines) < 2:
        return None, info

    # Outermost lines among the long ones; evaluate position at the image centre
    # so the ordering is not skewed by slope.
    def keep_long(lines):
        top = max(l[2] for l in lines)
        return [l for l in lines if l[2] >= 0.6 * top]
    hl = sorted(keep_long(hlines), key=lambda l: l[0] * w / 2 + l[1])
    vl = sorted(keep_long(vlines), key=lambda l: l[0] * h / 2 + l[1])
    top, bot, left, right = hl[0], hl[-1], vl[0], vl[-1]
    if abs(bot[0] * w / 2 + bot[1] - (top[0] * w / 2 + top[1])) < h * 0.3 or \
       abs(right[0] * h / 2 + right[1] - (left[0] * h / 2 + left[1])) < w * 0.3:
        return None, info
    quad = np.array([_intersect(top, left), _intersect(top, right),
                     _intersect(bot, right), _intersect(bot, left)], dtype=np.float32)
    info["angle_top_deg"] = float(np.degrees(np.arctan(top[0])))
    info["angle_left_deg"] = float(np.degrees(np.arctan(left[0])))
    return quad, info


def line_image(img: np.ndarray) -> np.ndarray:
    """Binary mask (255 = printed line) of long horizontal + vertical strokes.

    Handwriting and lighting gradients drop out, which is what makes lattice
    fitting work on photos where a global threshold does not.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                 max(15, (min(w, h) // 40) | 1), 15)
    hm = cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 60), 1)))
    vm = cv2.morphologyEx(binv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 40))))
    return cv2.bitwise_or(hm, vm)


def _norm_profile(prof: np.ndarray, win: int) -> np.ndarray:
    """Local max over +-win, scaled so a strong line is ~1."""
    p = prof.astype(np.float64)
    top = np.percentile(p[p > 0], 95) if np.any(p > 0) else 1.0
    p = np.clip(p / max(top, 1e-9), 0, 1)
    return cv2.dilate(p.reshape(1, -1), np.ones((1, 2 * win + 1))).ravel()


def _sample(prof: np.ndarray, pos: np.ndarray) -> np.ndarray:
    idx = np.round(pos).astype(int)
    ok = (idx >= 0) & (idx < len(prof))
    out = np.zeros(pos.shape)
    out[ok] = prof[idx[ok]]
    return out


def _snap(prof: np.ndarray, pos: float, rad: int) -> int:
    lo, hi = max(0, int(pos) - rad), min(len(prof), int(pos) + rad + 1)
    if hi <= lo:
        return int(round(pos))
    seg = prof[lo:hi]
    return lo + int(np.argmax(seg)) if seg.max() > 0.25 else int(round(pos))


def fit_grid(lines: np.ndarray, n_rows: int = 10, n_cols: int = 12) -> dict | None:
    """Fit the KNBSB batting grid as a lattice to a line mask (#16).

    Card structure used (measured on the verified Grizzlies scan):
      - name strip (left): each player row split in thirds; inning area: halves.
        So row boundaries are exactly the lines present in BOTH -> product score.
      - header row (p/2 tall, above player row 1): column boundaries only, no
        sub-column line; player rows: boundaries + one sub-column line.
    Returns {"row_tops", "row_bottoms", "col_lefts", "cell_size"} or None.
    """
    h, w = lines.shape
    m = (lines > 0).astype(np.float64)

    # ── Rows ──
    info = m[:, int(0.10 * w):int(0.28 * w)].sum(1)
    inning = m[:, int(0.45 * w):int(0.90 * w)].sum(1)
    p_lo, p_hi = h / 26, h / 10.5
    win = max(2, int(p_lo * 0.06))
    common = _norm_profile(info, win) * _norm_profile(inning, win)
    best = (-1.0, 0.0, 0.0)
    ks = np.arange(n_rows + 1)
    for p in np.arange(p_lo, p_hi, 0.5):
        y0s = np.arange(0, h - n_rows * p)
        if len(y0s) == 0:
            continue
        s = _sample(common, y0s[:, None] + ks[None, :] * p).sum(1)
        i = int(np.argmax(s))
        if s[i] > best[0]:
            best = (float(s[i]), float(y0s[i]), float(p))
    score_r, y0, p = best
    if score_r < n_rows * 0.5:
        return None
    # Snap on the raw inning-area profile: the dilated profile has flat tops,
    # so argmax would land win px left of the true line.
    inn_raw = _norm_profile(inning, 0)
    rad = max(2, int(p * 0.12))
    bounds = [_snap(inn_raw, y0 + k * p, rad) for k in ks]

    # ── Columns ──
    # Discrete rule instead of a profile score: the #/row-number columns left of
    # inning 1 are each ~half a cell wide, so they continue the half-pitch
    # pattern and fool any loose lattice fit. Column 1's left edge is the
    # leftmost boundary (line in header AND body) whose next half-step is a
    # sub-column line (body only) -- the narrow columns' lines all reach the header.
    top = bounds[0]
    head = m[max(0, int(top - 0.45 * p)):max(1, int(top - 0.1 * p)), :].sum(0)
    body = m[bounds[0]:bounds[-1], :].sum(0)
    gap = max(3, int(p * 0.03))
    B = cluster(raw_lines_from_projection(_norm_profile(body, 0), 0.3), gap)
    Hd = cluster(raw_lines_from_projection(_norm_profile(head, 0), 0.3), gap)
    half = [y - x for x, y in zip(B, B[1:]) if 0.35 * p <= y - x <= 0.65 * p]
    q = float(np.median(half)) if half else p / 2
    tol = 0.15 * q

    def near(x, pts):
        c = min(pts, key=lambda y: abs(y - x)) if pts else None
        return c if c is not None and abs(c - x) <= tol else None

    def col_start(b):
        s = near(b + q, B)
        return near(b, Hd) is not None and s is not None and near(s, Hd) is None

    x0 = next((b for b in B if col_start(b) and (n := near(b + 2 * q, B)) is not None and col_start(n)), None)
    if x0 is None:
        return None
    cols, hits = [x0], 0
    for _ in range(n_cols):
        t = cols[-1] + 2 * q
        if t >= w:
            break
        s = near(t, B)
        hits += s is not None
        cols.append(s if s is not None else int(round(t)))

    row_hits = sum(inn_raw[max(0, y - 1):y + 2].max() > 0.25 for y in bounds)
    return {"row_tops": bounds[:-1], "row_bottoms": bounds[1:], "col_lefts": cols,
            "cell_size": int(round(p)), "col_width": 2 * q,
            "score": round(min(row_hits / len(bounds), hits / max(1, len(cols) - 1)), 3)}


def rectify(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """Warp img so the detected grid quad becomes an axis-aligned rectangle.

    Returns (image, info). info["shift_px"] is the largest corner displacement
    the warp applies; info["rectified"] is False when no usable quad was found
    (the original image is returned).
    """
    quad, info = find_grid_quad(img)
    if quad is None:
        info["rectified"] = False
        return img, info
    tl, tr, br, bl = quad
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    # Anchor the target rectangle at the quad's centroid-aligned position so the
    # warped image keeps roughly the original framing and margins.
    cx, cy = quad.mean(axis=0)
    dst = np.array([[cx - width / 2, cy - height / 2], [cx + width / 2, cy - height / 2],
                    [cx + width / 2, cy + height / 2], [cx - width / 2, cy + height / 2]],
                   dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad, dst)
    h, w = img.shape[:2]
    out = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    info["shift_px"] = float(np.max(np.linalg.norm(dst - quad, axis=1)))
    info["rectified"] = True
    info["quad"] = quad.round(1).tolist()
    return out, info


# Warp only when it matters: clean flatbed scans need <6px (<5% of a cell),
# and leaving them untouched keeps their crops identical to past runs.
_RECTIFY_MIN_SHIFT_FRAC = 0.05
_MIN_FIT_SCORE = 0.8


def detect_grid_lattice(
    img_path: str,
    n_player_rows: int = 10,
    n_extra_rows: int = 2,
    rectified_out: str | None = None,
    debug_out: str | None = None,
):
    """Photo-capable grid detection (#16): rectify -> line mask -> lattice fit.

    Returns (row_tops, row_bottoms, extra_tops, extra_bottoms, col_lefts,
    cell_size, crop_path, info) — the first six exactly like
    probe_grid.detect_grid. crop_path is the image the coordinates refer to
    (rectified_out when the image was warped, else img_path). Returns None
    when the fit is not trustworthy, so the caller can fall back.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    warped, info = rectify(img)
    crop_path = img_path
    if info.get("rectified") and info["shift_px"] > _RECTIFY_MIN_SHIFT_FRAC * h / 13 and rectified_out:
        img, crop_path = warped, rectified_out
        cv2.imwrite(rectified_out, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        info["rectified"] = False
    print(f"Rectify: {'warped' if info['rectified'] else 'not needed'} "
          f"(corner shift {info.get('shift_px', 0):.1f}px, {info.get('n_h', 0)} H / {info.get('n_v', 0)} V lines)")

    g = fit_grid(line_image(img), n_rows=n_player_rows)
    if g is None or g["score"] < _MIN_FIT_SCORE:
        print(f"Lattice fit rejected (score={None if g is None else g['score']})")
        return None
    info["fit_score"] = g["score"]
    row_tops, row_bottoms, col_lefts, cell = g["row_tops"], g["row_bottoms"], g["col_lefts"], g["cell_size"]
    extra_tops = [row_bottoms[-1] + i * cell for i in range(n_extra_rows)]
    extra_bottoms = [t + cell for t in extra_tops]
    print(f"Lattice fit: score={g['score']}  cell={cell}px  col_width={g['col_width']:.1f}px")
    print(f"Row tops:    {row_tops}")
    print(f"Col lefts:   {col_lefts}")

    if debug_out:
        dbg = img.copy()
        for i, top in enumerate(row_tops):
            cv2.line(dbg, (0, top), (w, top), (0, 0, 255), 2)
            cv2.putText(dbg, f"P{i+1}", (5, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        cv2.line(dbg, (0, row_bottoms[-1]), (w, row_bottoms[-1]), (0, 0, 255), 2)
        for i, top in enumerate(extra_tops):
            cv2.line(dbg, (0, top), (w, top), (0, 128, 255), 2)
            cv2.putText(dbg, f"E{i+1}", (5, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 200), 1)
        for ci, x in enumerate(col_lefts):
            cv2.line(dbg, (x, 0), (x, h), (255, 0, 0), 2)
            cv2.putText(dbg, str(ci + 1), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
        cv2.imwrite(debug_out, dbg)
        print(f"Debug image -> {debug_out}")
    return row_tops, row_bottoms, extra_tops, extra_bottoms, col_lefts, cell, crop_path, info


if __name__ == "__main__":
    src = sys.argv[1]
    img = cv2.imread(src)
    if img is None:
        sys.exit(f"ERROR: cannot read {src}")
    out, info = rectify(img)
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in info.items() if k != "quad"})
    if len(sys.argv) > 2:
        cv2.imwrite(sys.argv[2], out)
