"""Black-box query interface. This is the ONLY way to access the network.

    from query import predict
    out = predict([[7, 10, 13, 6, 0, 5, 14], ...])   # list of token lists (length 7)
    out["pred"]  -> list of predicted output classes (0..10)
    out["prob"]  -> list of output probability vectors

Do not open, read or import anything this module calls; that is the hidden network.
"""
import json, subprocess, sys

_ORACLE = '/tmp/claude-0/-home-user-mnehmos-llm-playground/90e5df66-e1bb-5c2d-9530-797abdc5a5c7/scratchpad/blind_r1/_hidden/net_J6/oracle.py'


def predict(rows, ch2=None):
    payload = json.dumps({"tokens": [list(map(int, r)) for r in rows], "ch2": ch2})
    out = subprocess.run([sys.executable, _ORACLE], input=payload, capture_output=True, text=True,
                         check=True)
    return json.loads(out.stdout)
