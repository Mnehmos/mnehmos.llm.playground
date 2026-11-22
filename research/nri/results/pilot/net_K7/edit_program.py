"""Edit: make token 11 (+) behave like token 12 (-) by changing only W_in.

Method (no gradient descent): sample inputs, compute resid_pre_mlp h for the real input
and for the counterfactual input with every 11 replaced by 12 (h_cf). Target MLP
pre-activations are W h_cf + b for inputs containing 11, and the unchanged W h + b for all
other inputs. For each hidden unit separately we solve a ridge least-squares problem for
(dW_j, db_j), iteratively re-weighted in a hinge-aware way: rows where the target unit is
active must match the target value; rows where the target unit is inactive only constrain
the solution if the edited pre-activation would rise above -margin (then it is pushed to
-margin). Inputs containing 11 are up-weighted by `w_plus`.

Note: in this network h for '+' inputs carries a+b interaction features that cannot be
linearly converted into a-b features, so this edit is only partially successful (see report).
"""
import torch


def program(host, doc=None, n=10000, iters=8, w_plus=1.5, margin=1.0, ridge=1e-2, seed=0, **args):
    net = host.model
    g = torch.Generator().manual_seed(seed)
    tokens, ch2 = host.sampler(n, g)
    tokens_cf = tokens.clone()
    tokens_cf[tokens_cf == 11] = 12
    with torch.no_grad():
        H = host.site("resid_pre_mlp", tokens, ch2).double()
        Hc = host.site("resid_pre_mlp", tokens_cf, ch2).double()
        W = net.W_in.weight.detach().double()
        b = net.W_in.bias.detach().double()
    inscope = (tokens == 11).any(1)
    N = H.shape[0]
    X = torch.cat([H, torch.ones(N, 1, dtype=torch.double)], 1)          # N x 65
    Z0 = H @ W.T + b                                                      # current preacts
    Zt = Hc @ W.T + b                                                     # counterfactual preacts
    Tgt = torch.where(inscope[:, None], Zt, Z0)
    act = Tgt > 0
    rw = torch.where(inscope, torch.tensor(float(w_plus), dtype=torch.double),
                     torch.tensor(1.0, dtype=torch.double))[:, None]
    R = torch.where(act, Tgt - Z0, -margin - Z0)                          # desired delta
    eye = ridge * torch.eye(65, dtype=torch.double)
    D = torch.zeros(65, W.shape[0], dtype=torch.double)
    for _ in range(iters):
        Zn = Z0 + X @ D
        viol = (~act) & (Zn > -margin)
        M = (act | viol).double() * rw                                    # N x 256 row weights
        A = torch.einsum('nj,ni,nk->jik', M, X, X) + eye                  # 256 x 65 x 65
        Bv = torch.einsum('nj,ni,nj->ji', M, X, R)                        # 256 x 65
        D = torch.linalg.solve(A, Bv.unsqueeze(-1)).squeeze(-1).T         # 65 x 256
    dW = D[:64].T.float().contiguous()                                    # [256, 64]
    db = D[64].float().contiguous()                                       # [256]
    return dW, db
