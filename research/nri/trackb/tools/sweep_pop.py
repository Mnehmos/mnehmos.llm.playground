"""CPU-01A population sweep: W concurrent workers x T threads, each a 300-step mass_only prefix (6000-step horizon).
Reports aggregate updates/s (steady, excl. first 20 steps' wall via per-run first_step & timing) and end-to-end wall."""
import json, subprocess, sys, time, os, itertools
from pathlib import Path
W = Path(__file__).resolve().parent; R = str(W.parent.parent / "src"); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
STEPS = 300
paths = {"original": [], "exact_foreach": ["--fast", "--variant", "baseline", "--no-compile"],
         "numeric_uncompiled": ["--fast", "--no-compile"], "numeric_compiled": ["--fast"]}
configs = [("original", w, t) for w, t in ((1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (4, 1))] + \
          [(p, w, t) for p in ("exact_foreach", "numeric_uncompiled", "numeric_compiled") for w, t in ((1, 1), (4, 1))]
rows = []
env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(W.parent / "inductor_cache_fast"))
for trial in range(3):
    order = configs if trial % 2 == 0 else configs[::-1]
    for path, w, t in order:
        tag = f"{path}_{w}x{t}_r{trial}"
        t0 = time.time()
        ps = [subprocess.Popen([sys.executable, str(W / "fast_swap.py"), "--repo", R, "--zoo", str(OUT / tag), "--group", "mass_only",
                                "--seed", str(307 + i), "--steps", str(STEPS), "--threads", str(t)] + paths[path],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env) for i in range(w)]
        codes = [p.wait() for p in ps]
        wall = time.time() - t0
        cf = [json.load(open(OUT / tag / f"mass_only_s{307 + i}" / "config.json")) for i in range(w) if codes[i] == 0]
        loop = [c["train_seconds"] - c["first_step_seconds"] for c in cf]
        r = dict(path=path, workers=w, threads=t, trial=trial, status=codes, end_to_end_wall=wall,
                 aggregate_updates_per_s_steady=sum((STEPS - 1) / x for x in loop),
                 per_run_ms_per_update=[1e3 * x / (STEPS - 1) for x in loop],
                 first_step_s=[c["first_step_seconds"] for c in cf], peak_rss_mib=[c["peak_rss_mib"] for c in cf],
                 models_per_hour_6000_est=3600 * sum(1 / (x / (STEPS - 1) * 6000) for x in loop))
        rows.append(r); print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("per_run_ms_per_update",)}), flush=True)
        (OUT / "sweep.json").write_text(json.dumps(rows, indent=1))
