"""Exploratory factorial replay on unchanged Experiment 29 snapshots."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

root = Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root))
spec = importlib.util.spec_from_file_location("exp29",root/"experiment.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
rows=[]
for seed,start in itertools.product(e.SEEDS,(0,100)):
    file=root/"runs"/f"{seed}-B"/f"{start:04}.pt"
    saved=torch.load(file,weights_only=True)
    for use_c,use_f in itertools.product((False,True),repeat=2):
        s=e.Student(seed,"B")
        s.load_state_dict(saved["model"])
        opt=torch.optim.AdamW(s.parameters(),lr=.002)
        opt.load_state_dict(copy.deepcopy(saved["optimizer"]))
        gen=torch.Generator()
        gen.set_state(saved["rng"])
        torch.set_rng_state(saved["torch_rng"])
        initial=s.residual.W_U.weight[:13].detach().clone()
        for offset in range(1,101):
            c=e.cv_ids[torch.randint(len(e.cv_ids),(64,),generator=gen)]
            f=e.f_ids[torch.randint(len(e.f_ids),(96,),generator=gen)]
            lc=F.cross_entropy(s.convert(c),e.Y[c])
            lf=F.cross_entropy(s.forward_force(f),e.FY[f])
            opt.zero_grad()
            (int(use_c)*lc+int(use_f)*lf).backward()
            raw=float(s.residual.W_U.weight.grad[:13].norm())
            before=s.residual.W_U.weight[:13].detach().clone()
            opt.step()
            update=float((s.residual.W_U.weight[:13].detach()-before).norm())
            if offset in (1,5,10,20,50,100):
                scored=e.audit(s)
                rows.append(dict(seed=seed,start=start,offset=offset,conversion=use_c,force=use_f,
                                 authority=scored["intervention"]["intended"],
                                 practice=scored["conversion"]["practiced"],
                                 raw_readout_grad=raw,effective_readout_update=update,
                                 readout_drift=float((s.residual.W_U.weight[:13].detach()-initial).norm())))
        if use_c and use_f:
            expected=torch.load(file.with_name(f"{start+100:04}.pt"),weights_only=True)["model"]
            for k,v in s.state_dict().items():
                torch.testing.assert_close(v,expected[k],atol=0,rtol=0)
print(json.dumps(rows,indent=2))
