"""Write a review JSON from keyword args: python r.py STEP key=value ... (values are free text)."""
import json, sys
from pathlib import Path
step = sys.argv[1]
d = {}
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    d[k] = int(v) if k == "investigation_level" else v
p = Path(__file__).parent / f"review_{int(step):04d}.json"
p.write_text(json.dumps(d, indent=1))
print(p)
