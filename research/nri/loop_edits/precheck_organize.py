import sys, json; sys.path.insert(0, '.'); sys.path.insert(0, 'loop_edits')
import torch, run_loop as L
from common import organize_edit
st, s, meta = L.load_current_model()
m = st['model']
UP = ["attn1.qkv.weight", "attn1.qkv.bias", "attn1.o.weight", "attn1.o.bias", "attn2.qkv.weight", "attn2.qkv.bias", "attn2.o.weight", "attn2.o.bias"]
ALL = UP + ["W_in.weight", "W_in.bias", "W_out.weight", "W_out.bias"]
for c in json.loads(sys.argv[1]):
    params = {"UP": UP, "ALL": ALL}[c['p']]
    d = organize_edit(m, L.ctx(), params, **c['kw'])
    r = L.precheck(m, d, c.get('target', 'interaction'))
    rel = torch.cat([v.flatten() for v in d.values()]).norm().item() / L.model_norm(m)
    print(s, c['name'], f"rel {rel:.3f}", {k: round(v, 3) for k, v in r.items()}, flush=True)
