"""Deterministic instruments. All backward passes use the checkpoint copy."""
from collections import Counter
from itertools import combinations
import hashlib
import json

import torch
import torch.nn.functional as F

from sem import data as D, observe as O, train as T


def l2(vector):
    return float(vector.norm())


def cosine(left, right):
    denom = l2(left) * l2(right)
    return float(left.flatten() @ right.flatten()) / denom if denom else None


def record_step(state, batch, examples, twins, objectives, loss, parts):
    params = dict(state["model"].named_parameters())
    before = {k: p.detach().clone() for k, p in params.items()}
    optimizer = state["opt"]
    optimizer.step()
    groups = {id(p): g for g in optimizer.param_groups for p in g["params"]}
    tensors = {}
    for key, p in params.items():
        update = p.detach() - before[key]
        moments = optimizer.state.get(p, {})
        grad = p.grad
        lr = groups[id(p)]["lr"]
        tensors[key] = dict(raw_grad=l2(grad) if grad is not None else None,
                            actual_update=l2(update), update_over_weight=l2(update) / max(l2(before[key]), 1e-12),
                            cosine_grad_update=cosine(-grad, update) if grad is not None else None,
                            adam_m=l2(moments["exp_avg"]) if moments else None,
                            adam_v=l2(moments["exp_avg_sq"]) if moments else None, lr=lr)
    provenance = dict(examples=examples, twins=twins)
    state["interval"].append(dict(
        step=state["step"], loss=float(loss.detach()), components=parts, objectives=objectives,
        token_counts=dict(Counter(batch["tok"][batch["tok"] != D.PAD].tolist())),
        families=dict(Counter(e["fid"] for e in examples)),
        batch_sha256=hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
        tensors=tensors))


def weight_changes(current, prior, initial):
    report = {}
    for key, tensor in current.items():
        starts = {"previous": prior[key], "initial": initial[key]}
        row = {"shape": list(tensor.shape), "norm": l2(tensor)}
        for name, base in starts.items():
            delta = tensor - base
            row[name] = dict(l2=l2(delta), relative=l2(delta) / max(l2(base), 1e-12),
                             weight_cosine=cosine(tensor, base))
            if delta.ndim == 2:
                spectrum = torch.linalg.svdvals(delta)
                power = spectrum.square()
                row[name]["leading_sv"] = spectrum[:5].tolist()
                row[name]["top5_energy"] = float(power[:5].sum() / power.sum()) if power.sum() else None
        report[key] = row
    return report


def trace(model, vocab, examples):
    """Keep signed occurrence and parameter gradients for independent task losses."""
    observed = {}
    modules = dict(model.named_modules())
    names = ["E", "P", "attn1", "attn2", "attn3",
             "mlp1.fc", "mlp1", "mlp2.fc", "mlp2", "W_in", "W_out", "W_U"]
    hooks = [modules[name].register_forward_hook(
        lambda module, args, output, name=name: observed.__setitem__(name, output[0] if isinstance(output, tuple) else output))
        for name in names]
    parameters = dict(model.named_parameters())
    result, arrays, gradients = [], {}, []
    try:
        for ix, example in enumerate(examples):
            observed.clear()
            batch = D.encode_batch([example], vocab)
            graph = model.trunk(batch["tok"], ret=True)
            logits = model.head(graph["h"])
            # The readout uses functional linear, so capture its preactivation explicitly.
            observed["readout.pre_relu"] = graph["h"] @ model.W_in.weight.T + model.W_in.bias
            loss = F.cross_entropy(logits, batch["ans"])
            stages = dict(observed, **graph, logits=logits)
            # Compute this activation's gradient through the actual readout graph.
            post = observed["W_out"]
            observed_gradient = torch.autograd.grad(loss, post, retain_graph=True)[0]
            relu_gradient = (observed_gradient @ model.W_out.weight) * (stages["readout.pre_relu"] > 0)
            grads = torch.autograd.grad(loss, [*stages.values(), *parameters.values()], allow_unused=True)
            for name, value in stages.items():
                arrays[f"{ix}.activation.{name}"] = value.detach().contiguous()
            parameter_norms, activation_grads = {}, {}
            for name, grad in zip([*stages, *parameters], grads):
                if name == "readout.pre_relu":
                    grad = relu_gradient
                if grad is None:
                    continue
                kind = "gradient" if name in stages else "parameter_gradient"
                arrays[f"{ix}.{kind}.{name}"] = grad.detach().contiguous()
                (activation_grads if name in stages else parameter_norms)[name] = l2(grad)
            grad_e = arrays[f"{ix}.gradient.E"][0]
            tok = batch["tok"][0]
            aggregate = torch.zeros_like(model.E.weight).index_add_(0, tok, grad_e)
            torch.testing.assert_close(aggregate, arrays[f"{ix}.parameter_gradient.E.weight"])
            gradients.append(aggregate.flatten())
            result.append(dict(index=ix, example=example, loss=float(loss.detach()), logits=logits.detach()[0].tolist(),
                               occurrence_norms=grad_e.norm(dim=1).tolist(), token_ids=tok.tolist(),
                               activation_gradient_norms=activation_grads, parameter_gradient_norms=parameter_norms,
                               embedding_sum_verified=True))
    finally:
        for hook in hooks:
            hook.remove()
    align = {f"{i}:{j}": cosine(gradients[i], gradients[j]) for i, j in combinations(range(len(result)), 2)}
    return dict(examples=result, embedding_gradient_alignment=align), arrays


def objective_gradients(model, vocab, lesson):
    import random

    examples, twins, _ = D.lesson_batch(lesson, 32, random.Random(376))
    batch = D.encode_batch(examples, vocab)
    obj = dict(aux_role=0.5, iit=3., consistency=0.)
    obj.update(lesson["objectives"])
    _, losses = T.loss_on(model, batch, twins, obj, retain_parts=True)
    params = dict(model.named_parameters())
    vectors = {}
    for name, value in losses.items():
        g = torch.autograd.grad(value, tuple(params.values()), allow_unused=True, retain_graph=True)
        vectors[name] = {key: value.detach() if value is not None else torch.zeros_like(params[key])
                         for key, value in zip(params, g)}
    return dict(objectives=obj, panel=examples, twins=twins,
                tensor_norms={n: {k: l2(t) for k, t in v.items()} for n, v in vectors.items()},
                cosine_by_tensor={f"{a}:{b}": {k: cosine(vectors[a][k], vectors[b][k]) for k in params}
                                  for a, b in combinations(vectors, 2)}), {
                  f"{n}.{key}": g.contiguous() for n, ts in vectors.items() for key, g in ts.items()}


@torch.no_grad()
def causal(model, vocab, examples):
    batch = D.encode_batch(examples, vocab)
    h, _ = model.trunk(batch["tok"])
    M, A, S = O.canon(examples)
    donor = torch.randperm(len(h), generator=torch.Generator().manual_seed(288))
    expected = torch.stack((M*A*S, M[donor]*A*S, M*A[donor]*S), 1) % 13
    eligible = (expected.sort(1).values.diff(dim=1) != 0).all(1)
    cross = torch.tensor([examples[i]["fid"] != examples[d]["fid"] for i, d in enumerate(donor.tolist())])
    eligible &= cross
    assert eligible.any()
    end = model.head(h).argmax(1)
    result = dict(n=int(eligible.sum()), unchanged_on_target=float((end[eligible] == expected[eligible, 1]).float().mean()))
    for name, start in (("mass_block", 0), ("comparison_block", 16)):
        modified = h.clone()
        modified[:, start:start+16] = h[donor, start:start+16]
        pred = model.head(modified).argmax(1)[eligible]
        result[name] = {label: float((pred == expected[eligible, j]).float().mean())
                        for j, label in enumerate(("original", "mass", "acceleration"))}
    return result


def head_lineage(current, previous, prior_status):
    """Align only within a layer and withhold ambiguous identities."""
    import itertools

    out = {}
    for layer in range(1, 4):
        keys = [f"L{layer}.h{i}" for i in range(4)]
        if previous is None:
            out.update({n: dict(identity=n, status="baseline") for n in keys})
            continue
        a = F.normalize(torch.tensor([current[n] for n in keys]), dim=1)
        b = F.normalize(torch.tensor([previous[n] for n in keys]), dim=1)
        similarity = a @ b.T
        mapping = max(itertools.permutations(range(4)), key=lambda p: sum(float(similarity[i,j]) for i,j in enumerate(p)))
        for i, j in enumerate(mapping):
            sim = float(similarity[i, j])
            margin = sim - max(float(similarity[i, k]) for k in range(4) if k != j)
            prev = keys[j]
            old = prior_status[prev]
            eligible = sim >= .85 and margin >= .03
            streak = old.get("streak", 0) + 1 if eligible else 0
            out[keys[i]] = dict(previous=prev, similarity=sim, margin=margin, streak=streak,
                                identity=(old.get("identity") or prev) if streak >= 2 else None,
                                status="signature persistence" if streak >= 2 else "uncertain")
    return out
