import json
from pathlib import Path

cache = Path("images/_cache/cells/2026-06-07_almere")
grid = {}
for f in cache.glob("r??_c??.json"):
    parts = f.stem.split("_")
    ri = int(parts[0][1:])
    ci = int(parts[1][1:])
    grid[(ri, ci)] = json.loads(f.read_text(encoding="utf-8"))

# Correct player mapping — Lee shares row 4 with Dikkes, no separate row
players = ["Gelaudi","Bradwell","Spaandonk","Suares","Dikkes","Mabushi","Wisse","Romick","Trehy"]

print("ri,ci,player,inning,result,run,confidence,notes")
for ri in range(1, 10):   # 1-based (r01..r09)
    for ci in range(1, 10):  # 1-based (c01..c09)
        c = grid.get((ri, ci), {})
        result = c.get("result", "")
        run = c.get("run", False)
        confidence = c.get("confidence", "")
        notes = (c.get("notes") or "").replace(",", ";")
        print(f"{ri},{ci},{players[ri-1]},{ci},{result},{run},{confidence},{notes}")
