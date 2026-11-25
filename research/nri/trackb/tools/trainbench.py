"""Faithful run_swap.train_one loop with a pluggable encoder; timing or trace mode. Writes only to --out."""
import argparse, hashlib, json, os, sys, time, resource
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--repo", required=True); p.add_argument("--out", required=True)
p.add_argument("--variant", default="baseline"); p.add_argument("--group", default="mass_only")
p.add_argument("--seed", type=int, default=307); p.add_argument("--steps", type=int, default=100)
p.add_argument("--warmup", type=int, default=10); p.add_argument("--threads", type=int, default=1)
p.add_argument("--opt", default="default", choices=["default", "foreach", "fused", "single"])
p.add_argument("--compile", action="store_true"); p.add_argument("--nofill", action="store_true"); p.add_argument("--trace", action="store_true")
p.add_argument("--eval", action="store_true", help="include D.acc(n=4096) every 250 steps as in train_one")
a = p.parse_args()
T0 = time.perf_counter()
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]: os.environ[k] = str(a.threads)
sys.path.insert(0, str(Path(a.repo) / "research" / "nri")); sys.path.insert(0, str(Path(__file__).parent))
import torch, torch.nn.functional as F
import run_swap as S
import cand
D, T = S.D, S.T
torch.set_num_interop_threads(1); S.set_determinism(a.threads)
if a.nofill: torch.utils.deterministic.fill_uninitialized_memory = False
torch.manual_seed(a.seed)
m = S.TinyTyped(a.group, **S.MODEL_KW)
okw = dict(lr=S.LR, weight_decay=S.WD)
if a.opt == "foreach": okw["foreach"] = True
if a.opt == "fused": okw["fused"] = True
if a.opt == "single": okw["foreach"] = False
opt = torch.optim.AdamW(m.parameters(), **okw)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=S.LR, total_steps=S.STEPS, pct_start=0.1)
gb = torch.Generator().manual_seed(10000 + a.seed); gs = torch.Generator().manual_seed(20000 + a.seed)
var = S.VAR_OF.get(a.group); mk = D.mask(*S.BLOCK)
enc = lambda bt, st: cand.pair(m, D, bt, st, a.variant)
if a.compile:
    enc = torch.compile(enc, dynamic=False)
h = hashlib.sha256(); losses = []; tstart = None; tfirst = None
for step in range(1, a.steps + 1):
    if step == a.warmup + 1: tstart = time.perf_counter()
    lat_b = T.sample_latents(S.BATCH, gb); bb = T.build(lat_b)
    if var:
        lat_s = T.sample_latents(S.BATCH, gs); stok = T.build(lat_s).tokens
    else:
        stok = None
    hb, hs = enc(bb.tokens, stok)
    loss = F.cross_entropy(m.head_from_resid(hb), bb.y)
    if var:
        cf = D.counterfactual(lat_b, lat_s, var)
        loss = loss + S.lam_at(step, "warmup") * F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk), cf)
        if a.trace: h.update(cf.numpy().tobytes()); h.update(stok.numpy().tobytes())
    if a.trace: h.update(bb.tokens.numpy().tobytes())
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    losses.append(loss.item())
    if step == 1: tfirst = time.perf_counter() - T0
    if a.eval and step % 250 == 0: D.acc(m, n=4096)
el = time.perf_counter() - tstart
row = dict(vars(a), torch=torch.__version__, timed_seconds=el, ms_per_update=1e3 * el / (a.steps - a.warmup),
           updates_per_second=(a.steps - a.warmup) / el, first_step_seconds=tfirst, losses=losses,
           batch_digest=h.hexdigest() if a.trace else None,
           peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           cpu_seconds=time.process_time(), end_to_end_seconds=time.perf_counter() - T0)
Path(a.out).mkdir(parents=True, exist_ok=False)
if a.trace:
    torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), sched=sch.state_dict()), Path(a.out) / "state.pt")
(Path(a.out) / "metrics.json").write_text(json.dumps(row, indent=1))
print(json.dumps({k: row[k] for k in ["variant", "group", "opt", "compile", "ms_per_update", "peak_rss_mib"]}), flush=True)
