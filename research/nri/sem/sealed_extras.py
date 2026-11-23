"""Sealed-evaluation extras (run ONCE by run_sem.py final): causal localisation and persistent
editability on sealed force sets, paraphrase invariance on sealed phrasing families, binding at held-out
(sealed) values. Fitting data always comes from the training pool."""
from __future__ import annotations

from sem import data as D
from sem import observe as O


def extras(model, vocab, man, gen_eval):
    model.eval()
    sid, _ = gen_eval("sealed/force_id_trainfam", man["sealed"]["sealed/force_id_trainfam"]["seed"])
    snew, _ = gen_eval("sealed/force_newfam", man["sealed"]["sealed/force_newfam"]["seed"])
    sood, _ = gen_eval("sealed/force_ood_value", man["sealed"]["sealed/force_ood_value"]["seed"])
    fit = O.fit_pool()
    out = {"sealed_extra/iia_block_cross_family": round(O.iia_block(model, sid + snew, vocab), 4)}
    for kind in ("inv", "add1"):
        r = O.pooled_edit(model, fit, sid, vocab, kind)
        out[f"sealed_extra/edit_{kind}_intended"] = round(r["intended"], 4)
        out[f"sealed_extra/edit_{kind}_wrong"] = round(r["wrong"], 4)
    b = O.binding_probe(model, fit, sid, sood, vocab)
    out["sealed_extra/binding_probe_id"], out["sealed_extra/binding_probe_ood"] = round(b["id"], 4), round(b["ood"], 4)
    panel = D.paraphrase_panel(("pretrain", "train", "sealed"), seed=11)
    p = O.paraphrase_invariance(model, panel, vocab)
    out.update({f"sealed_extra/paraphrase_{k}": round(v, 4) for k, v in p.items()})
    return out
