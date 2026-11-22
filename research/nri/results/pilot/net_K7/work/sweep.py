import torch
exec(open('work/edit3.py').read().split("for name,Ht")[0])
sub=torch.nonzero(tr).flatten()[:10000]
for mg in [0.3,1.0,2.0]:
    for wp in [1,1.5,2,3]:
        print(mg,wp,[round(x,3) for x in evaluate(*hinge(Hc,wp,margin=mg,iters=8))])
