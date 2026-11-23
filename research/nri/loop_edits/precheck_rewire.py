import sys, json; sys.path.insert(0, '.'); sys.path.insert(0, 'loop_edits')
import torch, run_loop as L
from common import rewire_edit
st, s, meta = L.load_current_model()
m = st['model']
UP = ["attn1.qkv.weight", "attn1.qkv.bias", "attn1.o.weight", "attn1.o.bias", "attn2.qkv.weight", "attn2.qkv.bias", "attn2.o.weight", "attn2.o.bias"]
POST = ["W_in.weight", "W_in.bias", "W_out.weight", "W_out.bias"]
for c in json.loads(sys.argv[1]):
    post = POST if c['p'] in ("POST", "ALL") else []
    up = {k: torch.zeros_like(dict(m.named_parameters())[k]) for k in UP} if c['p'] in ("ALL", "UP") else {}
    d = rewire_edit(m, L.ctx(), up, post=post, also_upstream=bool(up), **c['kw'])
    d = {k: v for k, v in d.items() if v.abs().sum() > 0}
    for sc in c.get('scales', [1.0]):
        ds = {k: v * sc for k, v in d.items()}
        r = L.precheck(m, ds, c.get('target', 'edit_inv'))
        rel = torch.cat([v.flatten() for v in ds.values()]).norm().item() / L.model_norm(m)
        print(s, c['name'], 'x', sc, f"rel {rel:.3f}", {k: round(v, 3) for k, v in r.items()}, flush=True)
