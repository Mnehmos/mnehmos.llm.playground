"""CPU-01C/F candidate trainer: run_swap.train_one with an optional fast path. Fallback (fast=False) is the
original loop verbatim. Writes ONLY to --zoo (never the historical runs/zoo_swap).

fast path (numeric tier, not bitwise): layer-2 last-query with split Q, manual softmax, torch.compile of the
loss (fwd+bwd), foreach AdamW (bitwise-neutral). Sampling, RNG streams, schedules, eval cadence unchanged.
    python fast_swap.py --repo R --zoo OUT --group mass_only --seed 307 [--fast] [--threads 1]
"""
import argparse, json, os, sys, time, resource
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True); p.add_argument("--zoo", required=True)
    p.add_argument("--group", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--schedule", default="warmup"); p.add_argument("--fast", action="store_true")
    p.add_argument("--variant", default="manual+split"); p.add_argument("--no-compile", action="store_true")
    p.add_argument("--threads", type=int, default=1); p.add_argument("--steps", type=int, default=None,
                   help="stop early (prefix) while keeping the 6000-step scheduler horizon")
    a = p.parse_args()
    T0 = time.perf_counter()
    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]: os.environ[k] = str(a.threads)
    sys.path.insert(0, str(Path(a.repo) / "research" / "nri")); sys.path.insert(0, str(Path(__file__).parent))
    import torch, torch.nn.functional as F
    import run_swap as S
    import cand
    D, T = S.D, S.T
    run = Path(a.zoo) / f"{a.group}_s{a.seed}"
    if (run / "model.pt").exists():
        print("exists", run); return
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_interop_threads(1)
    S.set_determinism(a.threads)
    torch.manual_seed(a.seed)
    m = S.TinyTyped(a.group, **S.MODEL_KW)
    okw = dict(lr=S.LR, weight_decay=S.WD)
    if a.fast: okw["foreach"] = True
    opt = torch.optim.AdamW(m.parameters(), **okw)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=S.LR, total_steps=S.STEPS, pct_start=0.1)
    g_base = torch.Generator().manual_seed(10_000 + a.seed); g_src = torch.Generator().manual_seed(20_000 + a.seed)
    var = S.VAR_OF.get(a.group); mk = D.mask(*S.BLOCK)
    if a.fast:
        def L(bt, st, y, cf, lam):
            hb, hs = cand.pair(m, D, bt, st, a.variant)
            loss = F.cross_entropy(m.head_from_resid(hb), y)
            if st is not None:
                loss = loss + lam * F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk), cf)
            return loss
        lossf = L if a.no_compile else torch.compile(L, dynamic=False)
    steps = a.steps or S.STEPS
    cfg = dict(group=a.group, arm=a.group, condition=a.group, seed=a.seed, schedule=a.schedule, steps=S.STEPS,
               batch=S.BATCH, lr=S.LR, wd=S.WD, lambda_iit=S.LAMBDA if var else 0.0, declared=var, block=S.BLOCK,
               warmup_steps=S.WARMUP_STEPS if a.schedule == "warmup" else 0, n_params=m.n_params(),
               model_kw=S.MODEL_KW, torch=torch.__version__, encoder_forwards_per_step=2 if var else 1,
               trackb=dict(fast=a.fast, variant=a.variant if a.fast else "original", compiled=a.fast and not a.no_compile,
                           threads=a.threads, steps_run=steps))
    t0 = time.time(); tfirst = None
    with open(run / "log.jsonl", "w") as log:
        for step in range(1, steps + 1):
            lat_b = T.sample_latents(S.BATCH, g_base)
            bb = T.build(lat_b)
            if not a.fast:                                    # original loop, verbatim order
                _, hb = D.encode(m, bb.tokens)
                loss = F.cross_entropy(m.head_from_resid(hb), bb.y)
                if var:
                    lat_s = T.sample_latents(S.BATCH, g_src)
                    _, hs = D.encode(m, T.build(lat_s).tokens)
                    iit = F.cross_entropy(m.head_from_resid(hb * (1 - mk) + hs * mk), D.counterfactual(lat_b, lat_s, var))
                    loss = loss + S.lam_at(step, a.schedule) * iit
            else:
                if var:
                    lat_s = T.sample_latents(S.BATCH, g_src)
                    st, cf = T.build(lat_s).tokens, D.counterfactual(lat_b, lat_s, var)
                else:
                    st, cf = None, None
                loss = lossf(bb.tokens, st, bb.y, cf, torch.tensor(float(S.lam_at(step, a.schedule))))
            opt.zero_grad()
            loss.backward()
            if os.environ.get("DIAG") and a.fast and step in (1, 250, 500, 750, 1000, 1250):
                gc = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
                opt.zero_grad(); L(bb.tokens, st, bb.y, cf, torch.tensor(float(S.lam_at(step, a.schedule)))).backward()
                ge = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
                print("DIAG", step, "max grad diff", max(float((gc[n] - ge[n]).abs().max()) for n in ge),
                      "grad norm c/e", round(float(sum((g ** 2).sum() for g in gc.values()) ** .5), 4), round(float(sum((g ** 2).sum() for g in ge.values()) ** .5), 4),
                      "loss", round(loss.item(), 4), flush=True)
                for n, p in m.named_parameters():
                    if p.grad is not None: p.grad.copy_(gc[n])
            opt.step()
            sched.step()
            if step == 1: tfirst = time.time() - t0
            if step % 250 == 0 or step == S.STEPS:
                log.write(json.dumps(dict(step=step, loss=loss.item(), acc=D.acc(m, n=4096))) + "\n"); log.flush()
    cfg.update(acc_id=D.acc(m), acc_ood=D.acc(m, "ood"), train_seconds=round(time.time() - t0, 1),
               first_step_seconds=round(tfirst, 2), process_seconds=round(time.perf_counter() - T0, 1),
               cpu_seconds=round(time.process_time(), 1), peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    torch.save(m.state_dict(), run / "model.pt")
    print(f"[done] {run.name} fast={a.fast} id={cfg['acc_id']:.4f} ood={cfg['acc_ood']:.4f} train={cfg['train_seconds']}s", flush=True)

if __name__ == "__main__":
    main()
