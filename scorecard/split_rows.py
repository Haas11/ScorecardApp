#!/usr/bin/env python3
"""
split_rows.py — Slice a KNBSB scorecard into one image per player row.

Each output image contains:
  - The scorecard header (game info + inning numbers) at the top
  - One player's scoring row at the bottom

Usage:
  uv run python split_rows.py "path/to/scorecard.jpg"
  uv run python split_rows.py "path/to/scorecard.jpg" --out-dir images/rows
"""
from __future__ import annotations

from pathlib import Path

import click
import cv2
import numpy as np
from PIL import Image


def _detect_red_lines(rgb: np.ndarray, min_red_px: int = 80) -> list[int]:
    """
    Find horizontal red lines drawn on the scorecard (R high, G+B low).
    Returns the y-coordinate of the centre of each red-line band.
    """
    red_mask = (rgb[:, :, 0].astype(int) - rgb[:, :, 1].astype(int) > 60) & \
               (rgb[:, :, 0] > 130)
    red_per_row = red_mask.sum(axis=1).astype(float)

    # A "line row" has significantly more red than the background vertical lines
    # Background vertical lines score ~35 px/row; horizontal lines score 150–260 px.
    is_line = red_per_row > min_red_px

    ys = np.where(is_line)[0]
    if len(ys) == 0:
        return []

    # Cluster consecutive rows → single centre y per line
    dividers: list[int] = []
    cluster: list[int] = [int(ys[0])]
    for y in map(int, ys[1:]):
        if y - cluster[-1] <= 8:
            cluster.append(y)
        else:
            dividers.append(sum(cluster) // len(cluster))
            cluster = [y]
    dividers.append(sum(cluster) // len(cluster))
    return dividers


def _detect_all_lines(gray: np.ndarray, min_span_px: int = 50) -> list[int]:
    """
    Detect ALL horizontal lines (both player-dividers and sub-cell dividers).
    Uses adaptive thresholding to handle phone-photo vignetting.
    Returns clustered y-coordinates of detected lines.
    """
    # Adaptive threshold: pixels darker than their local neighbourhood
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        51, 5,
    )
    # Keep only horizontal runs >= min_span_px long
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_span_px, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    row_sums = h_lines.sum(axis=1).astype(float) / 255

    ys = np.where(row_sums > 20)[0]  # rows where a line segment exists
    if len(ys) == 0:
        return []

    # Cluster consecutive rows into single line positions
    dividers: list[int] = []
    cluster: list[int] = [int(ys[0])]
    for y in map(int, ys[1:]):
        if y - cluster[-1] <= 8:
            cluster.append(y)
        else:
            dividers.append(sum(cluster) // len(cluster))
            cluster = [y]
    dividers.append(sum(cluster) // len(cluster))
    return dividers


def _select_player_dividers(
    all_lines: list[int],
    img_height: int,
    num_players: int = 9,
) -> tuple[int, list[int]]:
    """
    From the full set of detected lines, select the header boundary and
    the num_players player-row boundaries.

    KNBSB scorecards have two horizontal lines per player row:
      - the sub-cell mid-divider (~half the row height)
      - the player-row boundary (full height)
    These appear as evenly-spaced pairs.  We discard sub-cell dividers by
    keeping only lines that are at least 50 px from the previous kept line.
    """
    # Filter: keep only lines at least 50 px apart (removes sub-cell dividers)
    player_level: list[int] = []
    prev = -100
    for y in all_lines:
        if y - prev >= 50:
            player_level.append(y)
            prev = y

    # Skip any thin-margin lines very close to the top of the image
    top_threshold = int(img_height * 0.07)
    player_level = [y for y in player_level if y >= top_threshold]

    if len(player_level) < 2:
        return 0, []

    # First surviving line = end of header row
    header_end = player_level[0]

    # Next num_players lines are the row boundaries
    row_bounds = player_level[1 : num_players + 1]

    return header_end, row_bounds


def split_scorecard(
    image_path: str | Path,
    out_dir: str | Path | None = None,
    num_players: int = 9,
    min_span_px: int = 50,
) -> list[Path]:
    """
    Detect horizontal player-row boundaries in *image_path*, then save each
    player row (prepended with the scorecard header) as a separate PNG.
    Returns the list of saved file paths.
    """
    img_path = Path(image_path)
    if out_dir is None:
        out_dir = img_path.parent / "rows"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_pil = Image.open(img_path).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    h, w = img_cv.shape

    all_lines = _detect_all_lines(img_cv, min_span_px)
    print(f"Raw lines detected: {len(all_lines)} at y = {all_lines}")

    header_end, row_bounds = _select_player_dividers(all_lines, h, num_players)
    print(f"Header ends at y={header_end}")
    print(f"Player row boundaries: {row_bounds}")

    if len(row_bounds) < num_players:
        print(
            f"Warning: only {len(row_bounds)} row boundaries found (expected {num_players}). "
            "Try --min-span-px with a smaller value if rows are missing."
        )

    if not row_bounds:
        return []

    # Header: everything above the first player row
    header = img_pil.crop((0, 0, w, header_end))

    # Build list of (y_start, y_end) for each player row
    row_starts = [header_end] + row_bounds[:-1]
    row_ends = row_bounds

    out_paths: list[Path] = []
    stem = img_path.stem

    for i, (y0, y1) in enumerate(zip(row_starts, row_ends), start=1):
        row_strip = img_pil.crop((0, y0, w, y1))

        # Stack header on top of player row so model sees inning numbers
        combined_h = header.height + row_strip.height
        combined = Image.new("RGB", (w, combined_h), (255, 255, 255))
        combined.paste(header, (0, 0))
        combined.paste(row_strip, (0, header.height))

        out_path = out_dir / f"{stem}_row{i:02d}.png"
        combined.save(out_path, "PNG")
        out_paths.append(out_path)
        print(f"  Row {i:2d}: y={y0}-{y1}  ({row_strip.height} px tall)  ->  {out_path.name}")

    crop_totals_strip(img_path, out_dir)
    return out_paths


def split_from_red_lines(
    image_path: str | Path,
    out_dir: str | Path | None = None,
    min_red_px: int = 80,
) -> list[Path]:
    """
    Split a scorecard using the red horizontal lines drawn on it as exact
    row boundaries.  The first red line is treated as the header/row-1 boundary;
    subsequent red lines are player-row boundaries.  The last player's row
    extends to the first line with few red pixels (end of scoring area) or
    to the same height as the other rows, whichever comes first.
    """
    img_path = Path(image_path)
    if out_dir is None:
        out_dir = img_path.parent / "rows"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_pil = Image.open(img_path).convert("RGB")
    rgb = np.array(img_pil)
    h, w = rgb.shape[:2]

    dividers = _detect_red_lines(rgb, min_red_px)
    print(f"Red-line dividers: {len(dividers)} at y = {dividers}")

    if len(dividers) < 2:
        print("Not enough red lines found.")
        return []

    # Estimate row height from median gap between dividers
    gaps = [dividers[i + 1] - dividers[i] for i in range(len(dividers) - 1)]
    row_h = int(np.median(gaps))
    print(f"Median row height: {row_h} px")

    # Header = everything above the first red line
    header_end = dividers[0]
    header = img_pil.crop((0, 0, w, header_end))

    # Player rows: between consecutive red lines; last player gets one extra estimated row
    boundaries = dividers + [dividers[-1] + row_h]

    out_paths: list[Path] = []
    stem = img_path.stem

    for i, (y0, y1) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        y1 = min(y1, h)  # don't exceed image height
        row_strip = img_pil.crop((0, y0, w, y1))

        combined_h = header.height + row_strip.height
        combined = Image.new("RGB", (w, combined_h), (255, 255, 255))
        combined.paste(header, (0, 0))
        combined.paste(row_strip, (0, header.height))

        out_path = out_dir / f"{stem}_row{i:02d}.png"
        combined.save(out_path, "PNG")
        out_paths.append(out_path)
        print(f"  Row {i:2d}: y={y0}-{y1}  ({row_strip.height} px)  ->  {out_path.name}")

    crop_totals_strip(img_path, out_dir, min_red_px)
    return out_paths


def crop_totals_strip(
    image_path: str | Path,
    out_dir: str | Path | None = None,
    min_red_px: int = 80,
) -> Path | None:
    """Crop the bottom team-totals band (the 'Totaal per inning' / 'Totaal'
    rows) and stack the inning header on top for column alignment. Saved as
    <stem>_totals.png. Returns the path, or None if the image is too small.
    """
    img_path = Path(image_path)
    if out_dir is None:
        out_dir = img_path.parent / "rows"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_pil = Image.open(img_path).convert("RGB")
    rgb = np.array(img_pil)
    h, w = rgb.shape[:2]

    dividers = _detect_red_lines(rgb, min_red_px)
    if len(dividers) >= 3:
        # The two bottom-most bands are the 'Totaal per inning' (2x2 cells:
        # TL=errors, TR=hits, BL=LOB, BR=runs) and the cumulative 'Totaal' row.
        # Start at the second-to-last red line so both are captured.
        header_end = dividers[0]
        start = dividers[-2]
    elif len(dividers) >= 2:
        gaps = [dividers[i + 1] - dividers[i] for i in range(len(dividers) - 1)]
        row_h = int(np.median(gaps))
        header_end = dividers[0]
        start = min(dividers[-1] + row_h, h - 1)
    else:
        # Fallback: assume a thin header and that totals sit in the bottom ~22%
        header_end = int(h * 0.05)
        start = int(h * 0.78)

    header = img_pil.crop((0, 0, w, header_end))
    band = img_pil.crop((0, start, w, h))
    combined = Image.new("RGB", (w, header.height + band.height), (255, 255, 255))
    combined.paste(header, (0, 0))
    combined.paste(band, (0, header.height))

    out_path = out_dir / f"{img_path.stem}_totals.png"
    combined.save(out_path, "PNG")
    print(f"  Totals strip: y={start}-{h}  ->  {out_path.name}")
    return out_path


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--out-dir", default=None, help="Directory for row crops (default: <image_dir>/rows/)")
@click.option("--num-players", default=9, show_default=True, help="Number of player rows to extract")
@click.option(
    "--min-span-px", default=50, show_default=True,
    help="Minimum horizontal pixel span for a line to be detected",
)
@click.option(
    "--red-lines", is_flag=True, default=False,
    help="Use red lines drawn on the image as exact row boundaries (more accurate for annotated images)",
)
def main(image_path: str, out_dir: str | None, num_players: int, min_span_px: int, red_lines: bool) -> None:
    """Slice a KNBSB scorecard image into one PNG per player row."""
    if red_lines:
        paths = split_from_red_lines(image_path, out_dir)
    else:
        paths = split_scorecard(image_path, out_dir, num_players, min_span_px)
    if paths:
        print(f"\n{len(paths)} row images saved to: {Path(paths[0]).parent}")
    else:
        print("\nNo rows saved — check the warnings above.")


if __name__ == "__main__":
    main()
