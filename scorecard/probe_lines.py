import cv2, numpy as np, pathlib

rows_dir = r'..\images\rows'
for rowname in ['row02', 'row04', 'row06']:
    matches = [m for m in pathlib.Path(rows_dir).glob(f'*{rowname}*.png')
               if not any(x in m.name for x in ['nodiag','clean','clahe','esrgan','debug'])]
    if not matches:
        continue
    img = cv2.imread(str(matches[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 10, 40, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=8,
                            minLineLength=40, maxLineGap=15)

    results = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = np.hypot(x2 - x1, y2 - y1)
        dx, dy = x2 - x1, y2 - y1
        angle = abs(np.degrees(np.arctan2(abs(dy), abs(dx)))) if abs(dx) > 0 else 90.0
        results.append((length, round(angle), x1, y1, x2, y2))
    results.sort(reverse=True)

    print(f"\n=== {rowname} (image {img.shape}) ===")
    print(f"  {'len':>6}  {'ang':>5}  coords")
    for length, angle, x1, y1, x2, y2 in results[:20]:
        print(f"  {length:6.0f}  {angle:5}  ({x1},{y1})-({x2},{y2})")

    # Generate a debug image: all lines longer than 70px in orange, rest in gray
    dbg = img.copy()
    for length, angle, x1, y1, x2, y2 in results:
        color = (0, 140, 255) if length >= 70 else (180, 180, 180)
        cv2.line(dbg, (x1, y1), (x2, y2), color, 2)
    out = str(pathlib.Path(rows_dir) / f"{rowname}_lines_debug.png")
    cv2.imwrite(out, dbg)
    print(f"  Saved {out}")
