"""Development-only curriculum and causal probes for written unit conversion."""
import random

import torch

from sem import data as D, lang as L


def lesson():
    return dict(task_weights={"convert": 1.0}, replay_weight=.2,
                families={"convert": ["C01"]}, objectives={"aux_role": .5, "iit": 0., "consistency": 0.},
                conversion_directions=[[1, 0]])


def examples(vocab, direction, values, family="C01", num="num", unit="full"):
    u, t = direction
    out = []
    for value in values:
        ast = dict(task="convert", dim="mass", v=value, u=u, t=t)
        if L.numeric_split(ast) != "train" or L.tuple_split(ast) == "sealed":
            continue
        row = D.make_example(ast, family, dict(num=num, unit=unit), random.Random(19))
        assert row["valid"] and all(token in vocab.stoi for token in row["tokens"]), row
        out.append(row)
    return out


@torch.no_grad()
def audit(model, vocab):
    sets = {
        "practiced": [e for e in examples(vocab,(1,0),range(13)) if L.tuple_split(e["ast"])=="train"],
        "new_values": [e for e in examples(vocab,(1,0),range(13)) if L.tuple_split(e["ast"])=="dev"],
        "new_frame": [e for e in examples(vocab,(1,0),range(13),"C02") if L.tuple_split(e["ast"])=="train"],
        "new_style": [e for e in examples(vocab,(1,0),range(13),"C01","word","abbr") if L.tuple_split(e["ast"])=="train"],
        "reverse": examples(vocab,(0,1),range(13)),
        "new_unit": examples(vocab,(2,0),range(9)),
    }
    result = {}
    for key, rows in sets.items():
        b = D.encode_batch(rows,vocab)
        pred = model(b["tok"]).argmax(1)
        result[key] = dict(correct=int((pred==b["ans"]).sum()), n=len(rows), predicted=pred.tolist(),
                           expected=b["ans"].tolist(), values=[r["ast"]["v"] for r in rows])
    rows = sets["practiced"]
    b = D.encode_batch(rows, vocab)
    tok, role = b["tok"], b["role"]
    c = model.trunk(tok,ret=True)
    i,j = torch.where(~torch.eye(len(rows), dtype=torch.bool))
    value = (role==L.ROLE_ID["src_val"]).long().argmax(1)
    unit = (role==L.ROLE_ID["src_unit"]).long().argmax(1)
    key = tok[i]!=D.PAD
    def finish(x):
        x = x+model.attn2(model.ln2a(x),key)
        x = x+model.mlp2(model.ln2m(x))
        x = x+model.attn3(model.ln3a(x),key)
        return model.head(x[:,-1]).argmax(1)
    assert torch.equal(finish(c["x1"][i]),model(tok)[i].argmax(1))
    a = torch.arange(len(i))
    for name, posts in (("value",[value]),("unit",[unit]),("value_and_unit",[value,unit])):
        patched=c["x1"][i].clone()
        for pos in posts:
            patched[a,pos[i]]=c["x1"][j,pos[j]]
        result["swap_"+name] = dict(intended=int((finish(patched)==b["ans"][j]).sum()),n=len(i))
    return result


def panel(vocab):
    rows = examples(vocab,(1,0),range(13))
    v = next(e["ast"]["v"] for e in rows if L.tuple_split(e["ast"])=="train")
    return (examples(vocab,(1,0),[v]) + examples(vocab,(1,0),[v],"C01","word","abbr")
            + examples(vocab,(1,0),[v],"C02"))
