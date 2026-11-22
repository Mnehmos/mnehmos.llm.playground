"""Edit W_in so that operator token 11 (+ mod 11) behaves like token 12 (- mod 11).

Closed-form, no gradient descent:
  1. sample inputs from the host distribution; compute residual h at resid_pre_mlp
  2. build counterfactual inputs with every 11 replaced by 12; compute h' there
  3. target MLP pre-activations: W_in h' + b for in-scope inputs (token 11 present),
     unchanged W_in h + b for the rest
  4. solve a (scope-upweighted) least-squares problem for the affine change
     [dW | db] acting on [h, 1] that best produces those pre-activations.
"""
import torch


def program(host, doc=None, n=20000, scope_weight=3.0, seed=0, ridge=100.0, **args):
    g = torch.Generator().manual_seed(seed)
    tokens, ch2 = host.sampler(n, g)
    tokens = torch.as_tensor(tokens).long()
    ch2 = torch.as_tensor(ch2).long()
    tokens2 = tokens.clone()
    tokens2[tokens2 == 11] = 12
    with torch.no_grad():
        H = host.site("resid_pre_mlp", tokens, ch2).float()
        H2 = host.site("resid_pre_mlp", tokens2, ch2).float()
        W = host.model.W_in.weight.detach().float()
        b = host.model.W_in.bias.detach().float()
        Z = H @ W.T + b
        Z2 = H2 @ W.T + b
        scope = (tokens == 11).any(1)
        delta = torch.where(scope[:, None], Z2 - Z, torch.zeros_like(Z))
        X = torch.cat([H, torch.ones(len(H), 1)], 1)
        w = torch.where(scope, float(scope_weight), 1.0)[:, None]
        G = (X * w).T @ X + ridge * torch.eye(X.shape[1])
        D = torch.linalg.solve(G, (X * w).T @ delta)  # [65, 256]
        dW = D[:-1].T.contiguous()  # [256, 64]
        db = D[-1].contiguous()     # [256]
    return dW, db
