"""Edit attempt: make the network treat operator token 11 ('+') as token 12 ('-').

Interpretation (see report.md): output = (a OP b) mod 11, OP = the single operator token
(11:'+', 12:'-', 13:'*'), a/b = the tokens immediately before/after it. Attention writes an
entangled (a, b, OP) code into the last-position residual h = resid_pre_mlp; the MLP reads it.

Method (closed-form weighted ridge least squares on sampled activations, no gradients):
  * sample inputs from the host distribution; for inputs containing token 11 build the
    counterfactual input with 11 -> 12 and read both residuals h, h'.
  * desired change of MLP pre-activations: delta = W_in (h' - h) on in-scope rows, 0 elsewhere.
  * solve [dW | db] = argmin sum_i w_i ||[dW | db][h_i; 1] - delta_i||^2 + ridge ||[dW|db]||^2.

HONEST CAVEAT: this edit largely FAILS at its goal. h('+') is nearly symmetric in (a, b)
(a-b is only ~50% linearly decodable from it) and the operator code is entangled with the
operand code (only ~20% of h('-') - h('+') is a constant vector), so no affine change of W_in
maps the '+' manifold onto '-' behaviour while fixing '-' and '*'. Measured trade-off (held-out):
ridge=1000,w=1: success ~0.11 / preserve ~0.99;  w=3: ~0.2 / ~0.85;  w=100: ~0.5 / ~0.45.
A gated logit correction through the 24 dead MLP neurons was also tried; it is infeasible because
the dead neurons' centred logit vectors lie in an open half-space (no positive null combination).
Default arguments favour preservation.
"""
import torch


def program(host, doc=None, n=20000, ridge=1000.0, w_in_scope=1.0, seed=0, **args):
    net = host.model
    g = torch.Generator().manual_seed(seed)
    tokens, ch2 = host.sampler(n, g)
    with torch.no_grad():
        h = host.site("resid_pre_mlp", tokens, ch2).double()
        has = (tokens == 11).any(1)
        tok2 = tokens.clone(); tok2[tok2 == 11] = 12
        h2 = host.site("resid_pre_mlp", tok2, ch2).double()
        W = net.W_in.weight.detach().double()
        delta = (h2 - h) @ W.T                      # [n, 256]
        delta[~has] = 0.0
        X = torch.cat([h, torch.ones(len(h), 1, dtype=h.dtype)], 1)   # [n, 65]
        w = torch.where(has, torch.tensor(float(w_in_scope), dtype=h.dtype),
                        torch.tensor(1.0, dtype=h.dtype))
        Xw = X * w[:, None]
        G = Xw.T @ X + ridge * torch.eye(X.shape[1], dtype=h.dtype)
        sol = torch.linalg.solve(G, Xw.T @ delta)    # [65, 256]
        dW = sol[:-1].T.contiguous()                 # [256, 64]
        db = sol[-1].contiguous()                    # [256]
    return dW.float(), db.float()
