"""CPU-01B: stage + op profile of the actual run_swap.train_one loop (pinned source, no zoo writes).
Mirrors train_one exactly (sampling order, encode, IIT, AdamW+OneCycle 6000-step horizon, D.acc every 250 steps).
"""
import argparse, json, os, sys, time, resource
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--repo", required=True); p.add_argument("--out", required=True)
p.add_argument("--group", default="mass_only"); p.add_argument("--seed", type=int, default=307)
p.add_argument("--steps", type=int, default=250); p.add_argument("--warmup", type=int, default=20)
p.add_argument("--threads", type=int, default=1); p.add_argument("--torchprof", type=int, default=20)
a = p.parse_args()
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]: os.environ[k] = str(a.threads)
sys.path.insert(0, str(Path(a.repo) / "research" / "nri"))
import torch, torch.nn.functional as F
import run_swap as S
D, T = S.D, S.T
torch.set_num_interop_threads(1); S.set_determinism(a.threads); torch.manual_seed(a.seed)
m = S.TinyTyped(a.group, **S.MODEL_KW)
opt = torch.optim.AdamW(m.parameters(), lr=S.LR, weight_decay=S.WD)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=S.LR, total_steps=S.STEPS, pct_start=0.1)
gb = torch.Generator().manual_seed(10000 + a.seed); gs = torch.Generator().manual_seed(20000 + a.seed)
var = S.VAR_OF.get(a.group); mk = D.mask(*S.BLOCK)
st = {k: 0.0 for k in ["sample_base", "build_base", "encode_base", "sample_src", "build_src", "encode_src",
                       "counterfactual", "head_loss", "backward", "optimizer", "eval_acc4096"]}
n_eval = 0
pc = time.perf_counter
def step_fn(step, timed):
    global n_eval
    t = pc(); lat_b = T.sample_latents(S.BATCH, gb); t1 = pc(); bb = T.build(lat_b); t2 = pc()
    _, hb = D.encode(m, bb.tokens); t3 = pc()
    loss = F.cross_entropy(m.head_from_resid(hb), bb.y); t4 = pc()
    d = dict(sample_base=t1 - t, build_base=t2 - t1, encode_base=t3 - t2, head_loss=t4 - t3)
    if var:
        t = pc(); lat_s = T.sample_latents(S.BATCH, gs); t1 = pc(); tok = T.build(lat_s).tokens; t2 = pc()
        _, hs = D.encode(m, tok); t3 = pc(); cf = D.counterfactual(lat_b, lat_s, var); t4 = pc()
        loss = loss + S.lam_at(step, "warmup") * F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk), cf); t5 = pc()
        d.update(sample_src=t1 - t, build_src=t2 - t1, encode_src=t3 - t2, counterfactual=t4 - t3)
        d["head_loss"] += t5 - t4
    t = pc(); opt.zero_grad(); loss.backward(); t1 = pc(); opt.step(); sch.step(); t2 = pc()
    d.update(backward=t1 - t, optimizer=t2 - t1)
    if step % 250 == 0:
        t = pc(); loss.item(); D.acc(m, n=4096); d["eval_acc4096"] = pc() - t; n_eval += 1
    if timed:
        for k, v in d.items(): st[k] += v
t0 = None
for step in range(1, a.steps + 1):
    if step == a.warmup + 1: t0 = pc()
    step_fn(step, step > a.warmup)
wall = pc() - t0
nt = a.steps - a.warmup
row = dict(group=a.group, seed=a.seed, threads=a.threads, steps=a.steps, warmup=a.warmup, timed_steps=nt, wall=wall,
           ms_per_update=1e3 * wall / nt, stage_ms_per_update={k: 1e3 * v / nt for k, v in st.items()},
           stage_share={k: v / wall for k, v in st.items()}, eval_calls=n_eval, torch=torch.__version__)
# op-level profile (separate, after timing)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU]) as prof:
    for s in range(a.steps + 1, a.steps + 1 + a.torchprof): step_fn(s, False)
tab = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
row["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
Path(a.out).mkdir(parents=True, exist_ok=False)
(Path(a.out) / "profile.json").write_text(json.dumps(row, indent=2)); (Path(a.out) / "ops.txt").write_text(tab)
print(json.dumps({k: row[k] for k in ["group", "ms_per_update", "peak_rss_mib"]}), {k: round(v, 3) for k, v in row["stage_ms_per_update"].items()})
print(tab)
