"""Edit: wherever token 11 ('+') occurs, behave as if it were 12 ('-'); keep all else.

Closed-form (least squares, no gradient descent). For every sampled input x we compute the
counterfactual x' = x with 11 -> 12, and the activations h = site(x), h' = site(x').
We want new pre-activations  (W+dW) h + (b+db)  ==  W h' + b   on in-scope inputs (x != x')
                                                ==  W h + b    elsewhere,
i.e. dW h + db = W (h' - h) * [in scope].  This is a linear regression from [h;1] (65 dims)
to the 256 pre-activation targets, solved by (weighted) ridge least squares. Several variants
(ridge strength, out-of-scope weight, ReLU-aware per-neuron masking) are scored on held-out
samples through host.model.head_from_resid, and the best one is returned.
"""
import torch


def _feats(h):
    return torch.cat([h, torch.ones(h.shape[0], 1, dtype=h.dtype)], 1)


def _solve(X, Y, w, lam):
    # weighted ridge: min sum_i w_i ||M x_i - y_i||^2 + lam ||M||^2  ;  M: [65 -> 256]
    Xw = X * w[:, None]
    A = X.T @ Xw + lam * torch.eye(X.shape[1], dtype=X.dtype)
    B = Xw.T @ Y
    return torch.linalg.solve(A, B)  # [65, 256]


def _solve_hinge(X, C, T, w, lam, iters=6):
    """Per-neuron ReLU-aware fit (semismooth Newton / active-set least squares, closed-form steps).
    For neuron j and sample i, new pre-activation n_ij = C_ij + X_i . q_j.
      if target T_ij > 0 : want n_ij == T_ij        (squared error)
      if target T_ij <= 0: want n_ij <= 0           (squared hinge; exact value irrelevant after ReLU)
    """
    N, k = X.shape
    q = _solve(X, T - C, w, lam)  # plain LS start  [k, 256]
    I = torch.eye(k, dtype=X.dtype)
    pos = T > 0
    Y = torch.where(pos, T - C, -C)
    for _ in range(iters):
        Nn = C + X @ q
        act = pos | (Nn > 0)
        Wm = act.to(X.dtype) * w[:, None]
        for j in range(T.shape[1]):
            Xw = X * Wm[:, j:j + 1]
            q[:, j] = torch.linalg.solve(X.T @ Xw + lam * I, Xw.T @ Y[:, j])
    return q


def _counterfactuals(tok):
    """Two ways to express 'treat + as -' as a target activation:
       'swap'  : replace token 11 by 12 (literal counterfactual input)
       'negb'  : keep '+', replace the right operand b by (-b) mod 11, since a - b = a + (-b) (mod 11);
                 this asks the network's own '+' circuit to produce the '-' answer, which is often a
                 much more linear change of the activation than switching the operator."""
    swap = tok.clone(); swap[swap == 11] = 12
    negb = tok.clone()
    for k in range(1, tok.shape[1] - 1):
        m = (tok[:, k] == 11) & (tok[:, k + 1] <= 10)
        negb[m, k + 1] = (-tok[m, k + 1]) % 11
    return {"swap": swap, "negb": negb}


def program(host, doc=None, n=16000, seed=0, **args):
    g = torch.Generator().manual_seed(seed)
    W = host.model.W_in.weight.detach().double()
    b = host.model.W_in.bias.detach().double()

    def gather(n):
        tok, ch2 = host.sampler(n, g)
        tok = torch.as_tensor(tok).long()
        scope = (tok == 11).any(1)
        with torch.no_grad():
            h = host.site("resid_pre_mlp", tok, ch2).detach().double()
            cf = {k: host.site("resid_pre_mlp", v, ch2).detach().double() for k, v in _counterfactuals(tok).items()}
        return tok, ch2, h, cf, scope

    tok, ch2, h, cf, scope = gather(n)
    vt, vch2, vh, vcf, vscope = gather(max(4000, n // 4))
    X = _feats(h)
    P_cur = h @ W.T + b

    with torch.no_grad():
        dt = host.model.W_in.weight.dtype
        def logits(hh, dW, db):
            return host.model.head_from_resid(hh.to(dt), (W + dW).to(dt), (b + db).to(dt))
        Z, z = torch.zeros_like(W), torch.zeros_like(b)
        base_v = logits(vh, Z, z).argmax(-1)
        targ_v = logits(vcf["swap"], Z, z).argmax(-1)   # what the net outputs with 11 -> 12

        def score(M):
            p = logits(vh, M[:64].T, M[64]).argmax(-1)
            succ = (p[vscope] == targ_v[vscope]).double().mean().item() if vscope.any() else 1.0
            pres = (p[~vscope] == base_v[~vscope]).double().mean().item() if (~vscope).any() else 1.0
            return succ, pres

        s = X.std(0).mean().item() ** 2 * len(X)
        best = None
        results = []
        for kind in ("negb", "swap"):
            P_tgt = torch.where(scope[:, None], cf[kind] @ W.T + b, P_cur)
            for wo in (0.5, 2.0):
                w = torch.where(scope, torch.ones(len(X), dtype=X.dtype), torch.full((len(X),), wo, dtype=X.dtype))
                for lam in (1e-6, 1e-3):
                    for solver in ("ls", "hinge"):
                        M = _solve_hinge(X, P_cur, P_tgt, w, lam * s) if solver == "hinge" else _solve(X, P_tgt - P_cur, w, lam * s)
                        succ, pres = score(M)
                        results.append((kind, wo, lam, solver, round(succ, 4), round(pres, 4)))
                        key = (min(succ, 0.9 if pres >= 0.97 else pres * 0.9 / 0.97), succ + pres)
                        if best is None or key > best[0]:
                            best = (key, M, results[-1])
    for r in results:
        print("[edit_program] candidate", r)
    _, M, info = best
    print("[edit_program] chosen", info)
    dW = M[:64].T.contiguous().to(host.model.W_in.weight.dtype)
    db = M[64].contiguous().to(host.model.W_in.bias.dtype)
    return dW, db
