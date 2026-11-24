"""Score development-only conversion strata and constant-output references."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from sem import data as D, train as T


@torch.no_grad()
def main(checkpoint):
    state = T.load_state(checkpoint)
    model = state["model"].eval()
    records = {}
    for file in sorted((checkpoint.parent/"development/dev").glob("convert*.jsonl")):
        rows = D.read_jsonl(file)
        labels, predictions = [], []
        for part in range(0,len(rows),64):
            b = D.encode_batch(rows[part:part+64],state["vocab"])
            labels.extend(b["ans"].tolist())
            predictions.extend(model(b["tok"]).argmax(1).tolist())
        groups = defaultdict(list)
        for ex, label, pred in zip(rows,labels,predictions):
            ast = ex["ast"]
            for key in (ast["dim"],f'{ast["dim"]}:{ast["u"]}->{ast["t"]}',"all"):
                groups[key].append((label,pred))
        records[file.stem] = {key:dict(n=len(pairs),hits=sum(a==b for a,b in pairs),
            best_constant=max(Counter(a for a,b in pairs).values()),
            most_frequent_prediction=Counter(b for a,b in pairs).most_common(1)[0])
            for key,pairs in groups.items()}
    return records


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])),indent=2))
