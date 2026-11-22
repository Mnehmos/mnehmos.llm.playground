"""Edit: make the network treat token 11 ('+', mod-11 addition) as token 12 ('-', mod-11 subtraction).

Closed-form method (no gradient descent):
  1. Sample inputs, build counterfactual copies with every 11 replaced by 12.
  2. Read h = resid_pre_mlp for originals and counterfactuals.
  3. Target change of the MLP pre-activation on '+' inputs: W_in (h_cf - h); on all other inputs: 0.
  4. Solve a weighted ridge regression  [dW | db] @ [h; 1] ~= target  (one linear solve).
  4b. Refine with a ReLU-aware per-neuron reweighted least squares (pairs whose target is
      negative and already negative after the edit are down-weighted), a few closed-form iterations.
  5. Pick a candidate, scale / SVD rank on a held-out sample, scored by the network's own head
     (success on '+' inputs vs. the counterfactual output; preservation elsewhere).
"""
import torch


def _as_long(t):
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    return t.long()


@torch.no_grad()
def program(host, doc=None, n_fit=12000, n_val=4000, ridge=1e-3, other_weight=1.0, seed=0, irls_iters=4, other_weights=(0.1, 0.3, 1.0), **args):
    W = host.model.W_in.weight.detach()
    b = host.model.W_in.bias.detach()
    dev, dt = W.device, W.dtype
    g = torch.Generator().manual_seed(seed)

    def batch(n):
        tok, ch2 = host.sampler(n, g)
        tok = _as_long(tok)
        cf = tok.clone()
        cf[cf == 11] = 12
        ch2_cf = ch2.clone() if torch.is_tensor(ch2) else ch2
        h = host.site("resid_pre_mlp", tok, ch2).detach().to(dev, torch.float64)
        hc = host.site("resid_pre_mlp", cf, ch2_cf).detach().to(dev, torch.float64)
        plus = (tok == 11).any(1).to(dev)
        return h, hc, plus

    def logits(h, Wm, bm, bs=2048):
        out = []
        for i in range(0, h.shape[0], bs):
            out.append(host.model.head_from_resid(h[i:i + bs].to(dt), Wm.to(dt), bm.to(dt)).float())
        return torch.cat(out)

    # ---- fit ----
    h, hc, plus = batch(n_fit)
    W64, b64 = W.to(torch.float64), b.to(torch.float64)
    target = (hc - h) @ W64.T                      # [n, 256] desired change in pre-activation
    target[~plus] = 0.0
    X = torch.cat([h, torch.ones(h.shape[0], 1, dtype=torch.float64, device=dev)], 1)  # [n, 65]
    w = torch.where(plus, torch.ones_like(h[:, 0]), torch.full_like(h[:, 0], other_weight))
    Xw = X * w[:, None]
    A = X.T @ Xw
    A = A + ridge * torch.trace(A) / A.shape[0] * torch.eye(A.shape[0], dtype=torch.float64, device=dev)
    D = torch.linalg.solve(A, Xw.T @ target).T     # [256, 65]

    # ---- candidate edits ----
    cands = []
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    for r in sorted(set([r for r in (1, 2, 3, 4, 8, 16, 32, 65) if r <= S.numel()])):
        cands.append(("ls", r, (U[:, :r] * S[:r]) @ Vh[:r]))

    # ReLU-aware per-neuron iteratively reweighted least squares (closed-form solves, no gradients):
    # a (sample, neuron) pair whose target pre-activation is <= 0 only needs the edited value to stay <= 0,
    # so once it does, it is down-weighted and the neuron's fit concentrates on pairs that matter.
    z0 = X @ torch.cat([W64, b64[:, None]], 1).T            # [n, 256] current pre-activations
    zt = z0 + target                                       # desired pre-activations
    I65 = torch.eye(X.shape[1], dtype=torch.float64, device=dev)
    for ow in other_weights:
        w_ow = torch.where(plus, torch.ones_like(h[:, 0]), torch.full_like(h[:, 0], ow))
        Dn = D.clone()
        for it in range(irls_iters):
            ze = z0 + X @ Dn.T
            sat = (zt <= 0) & (ze <= 0)
            wt = torch.where(sat, torch.full_like(ze, 0.02), torch.ones_like(ze)) * w_ow[:, None]
            newD = torch.empty_like(Dn)
            for j in range(W64.shape[0]):
                Xw_j = X * wt[:, j:j + 1]
                A_j = X.T @ Xw_j
                A_j = A_j + ridge * torch.trace(A_j) / A_j.shape[0] * I65
                newD[j] = torch.linalg.solve(A_j, Xw_j.T @ target[:, j])
            Dn = newD
            if it >= 1:
                cands.append(("irls_ow%g_it%d" % (ow, it), X.shape[1], Dn.clone()))
    scales = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)

    hv, hcv, plusv = batch(n_val)
    base = logits(hv, W64, b64).argmax(1)
    want = logits(hcv, W64, b64).argmax(1)         # network's own output on the 11->12 input
    want = torch.where(plusv, want, base)

    best = None
    for name, r, Dr in cands:
        for a in scales:
            dW, db = a * Dr[:, :-1], a * Dr[:, -1]
            pred = logits(hv, W64 + dW, b64 + db).argmax(1)
            ok = pred == want
            succ = ok[plusv].float().mean().item() if plusv.any() else 1.0
            pres = ok[~plusv].float().mean().item() if (~plusv).any() else 1.0
            score = min(succ, pres) + 0.5 * (succ + pres) - 1e-4 * r
            if best is None or score > best[0]:
                best = (score, (name, r), a, succ, pres, dW, db)
    _, r, a, succ, pres, dW, db = best
    try:
        print(f"[edit_program] chosen rank={r} scale={a} val_success={succ:.4f} val_preserve={pres:.4f}")
    except Exception:
        pass
    return dW.to(dev, dt).contiguous(), db.to(dev, dt).contiguous()
