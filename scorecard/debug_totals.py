import sys, cv2, numpy as np
sys.path.insert(0, ".")
from probe_grid import detect_grid

img_path = "../images/scans/2026-04-12 - Thamen (Home).jpg"
row_tops, row_bottoms, extra_tops, extra_bottoms, col_lefts, cell_size = detect_grid(
    img_path, n_player_rows=10, n_inning_cols=12
)
img = cv2.imread(img_path)
print(f"\nextra_tops={extra_tops}  extra_bottoms={extra_bottoms}")
print(f"col_lefts={col_lefts}")
print()
for row_idx, (y1, y2) in enumerate(zip(extra_tops, extra_bottoms)):
    print(f"Extra row {row_idx+1} (y={y1}..{y2}):")
    for ci in range(12):
        x1 = max(0, col_lefts[ci])
        x2 = col_lefts[ci+1] if ci+1 < len(col_lefts) else col_lefts[-1]
        crop = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        frac = int(np.sum(gray < 128)) / gray.size
        print(f"  col{ci+1:2d}  dark_frac={frac:.4f}")
