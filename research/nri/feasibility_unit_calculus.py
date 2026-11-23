"""Feasibility probe (exploratory, not registered) on existing undeclared ('neither') networks.
Q1 where: which quantities are linearly decodable at which role-aligned position and layer.
Q2 calculator: are value representations organised by the multiplicative group (discrete log)
   or by the additive group (value order)? Character power spectrum of class means.
Q3 label relationship: does switching a unit label act like multiplying the canonical value
   by the scale ratio? (unit-trade invariance of representations)"""
import sys, json, math, statistics as st
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import torch
import run_dimensional as D
from nri import dimtask as T
from concurrent.futures import ProcessPoolExecutor

P = 13
G = 2  # primitive root mod 13
LOG = {pow(G, k, P): k for k in range(12)}


def sites(m, lat):
    b = T.build(lat)
    tok = b.tokens
    z = torch.zeros_like(tok)
    with torch.no_grad():
        x0 = m.embed(tok, z)
        x1 = x0 + m.attn1(x0)
        x2 = x1 + m.attn2(x1)
        h = x2[:, -1]
        r = h + m.W_out(torch.relu(m.W_in(h)))
    first = lat["order"] == 0
    idx = lambda a, b_: torch.where(first, torch.full_like(lat["order"], a), torch.full_like(lat["order"], b_))
    roles = {"mass_val": idx(0, 3), "mass_unit": idx(1, 4), "acc_val": idx(3, 0), "acc_unit": idx(4, 1),
             "out_unit": torch.full_like(lat["order"], 6), "query": torch.full_like(lat["order"], 7)}
    ar = torch.arange(len(tok))
    out = {}
    for L, x in (("L0", x0), ("L1", x1), ("L2", x2)):
        for role, pos in roles.items():
            out[f"{L}:{role}"] = x[ar, pos]
    out["L3:query(post-MLP)"] = r
    return out


def probe_acc(X, y, ntr):
    """Ridge one-hot linear probe; accuracy on held-out rows."""
    Xa = torch.cat([X, torch.ones(len(X), 1)], 1).double()
    Y = torch.nn.functional.one_hot(y, int(y.max()) + 1).double()
    A = Xa[:ntr]
    W = torch.linalg.lstsq(A, Y[:ntr], driver="gelsd").solution
    return ((Xa[ntr:] @ W).argmax(1) == y[ntr:]).float().mean().item()


def char_spectrum(means, kind):
    """Share of class-mean variance (nonzero classes 1..12) at each frequency of the
    multiplicative characters (angle = 2*pi*f*log_g(x)/12) or additive characters
    (angle = 2*pi*f*x/13)."""
    xs = list(range(1, 13))
    Mu = means[xs] - means[xs].mean(0)
    tot = (Mu ** 2).sum()
    spec = []
    nf = 6
    for f in range(1, nf + 1):
        if kind == "mult":
            ang = torch.tensor([2 * math.pi * f * LOG[x] / 12 for x in xs])
        else:
            ang = torch.tensor([2 * math.pi * f * x / 13 for x in xs])
        B = torch.stack([torch.cos(ang), torch.sin(ang)], 1)
        Q, _ = torch.linalg.qr(B - B.mean(0))
        spec.append(((Q.T @ Mu) ** 2).sum().item() / tot.item())
    return spec


def run(path):
    torch.manual_seed(0)
    m, cfg = D.load(path)
    lat = T.sample_latents(20000, torch.Generator().manual_seed(4242))
    M, A, S = T.canon(lat)
    labels = {"m(raw)": lat["m"], "mu(label)": lat["mu"], "M(canon)": M, "a(raw)": lat["a"],
              "au(label)": lat["au"], "A(canon)": A, "S": lat["fu"], "M*A": M * A % P, "F": T.target(M, A, S)}
    X = sites(m, lat)
    ntr = 15000
    res = {"model": cfg["group"] + str(cfg["seed"]), "probe": {}, "spectrum": {}, "trade": {}}
    for s, x in X.items():
        res["probe"][s] = {k: round(probe_acc(x, y, ntr), 3) for k, y in labels.items()}
    # calculator geometry: class means of the canonical / product / answer at key sites
    for s, var in (("L1:mass_unit", "M(canon)"), ("L2:query", "M(canon)"), ("L2:query", "A(canon)"),
                   ("L2:query", "M*A"), ("L3:query(post-MLP)", "F")):
        y = labels[var]
        means = torch.stack([X[s][y == c].mean(0) for c in range(P)])
        res["spectrum"][f"{s}|{var}"] = {"mult": [round(v, 3) for v in char_spectrum(means, "mult")],
                                         "add": [round(v, 3) for v in char_spectrum(means, "add")]}
    # unembedding rows for answers 1..12
    WU = m.W_U.weight.detach()
    res["spectrum"]["W_U|F"] = {"mult": [round(v, 3) for v in char_spectrum(WU, "mult")],
                                "add": [round(v, 3) for v in char_spectrum(WU, "add")]}
    # unit-trade invariance: same canonical M via different (value, unit) routes. Share of
    # variance at each site explained by canonical M vs by the raw (value, unit) pair
    for s in ("L1:mass_val", "L1:mass_unit", "L2:mass_unit", "L2:query"):
        x = X[s] - X[s].mean(0)
        tot = (x ** 2).sum()
        def expl(key):
            k = key
            means = {}
            out = torch.zeros_like(x)
            for c in k.unique():
                sel = k == c
                out[sel] = x[sel].mean(0)
            return ((out ** 2).sum() / tot).item()
        res["trade"][s] = {"canonical M": round(expl(M), 3),
                           "raw (m, unit)": round(expl(lat["m"] * 3 + lat["mu"]), 3)}
    return res


if __name__ == "__main__":
    runs = sys.argv[1:]
    with ProcessPoolExecutor(4) as ex:
        R = list(ex.map(run, runs))
    json.dump(R, open("results/unitcalc_feasibility/probe.json", "w"), indent=1)
    # summaries
    sites_ = list(R[0]["probe"])
    keys = list(R[0]["probe"][sites_[0]])
    print("PROBE ACCURACY (mean over networks); chance ~0.08 for 13-way, 0.33 for labels")
    print("site".ljust(22), " ".join(k[:9].rjust(9) for k in keys))
    for s in sites_:
        print(s.ljust(22), " ".join(f"{st.mean(r['probe'][s][k] for r in R):9.2f}" for k in keys))
    print("\nCHARACTER SPECTRUM of class means (share of variance per frequency 1..6)")
    for s in R[0]["spectrum"]:
        mu_ = [round(st.mean(r["spectrum"][s]["mult"][f] for r in R), 2) for f in range(6)]
        ad = [round(st.mean(r["spectrum"][s]["add"][f] for r in R), 2) for f in range(6)]
        print(s.ljust(34), "mult", mu_, " add", ad)
    print("\nUNIT TRADE: share of variance explained by canonical M vs raw (value, unit)")
    for s in R[0]["trade"]:
        print(s.ljust(16), {k: round(st.mean(r["trade"][s][k] for r in R), 2) for k in R[0]["trade"][s]})
