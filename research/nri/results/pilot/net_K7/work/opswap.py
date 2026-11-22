import torch
exec(open('work/edit_try.py').read().split('for lam in')[0])
L=T.tolist()
a=torch.tensor([t[t.index(11)-1] if 11 in t else 0 for t in L]); bb=torch.tensor([t[t.index(11)+1] if 11 in t else 0 for t in L])
opid=torch.full((len(T),),-1)
for o in (11,12,13): opid[(T==o).any(1)]=o-11
mu=torch.stack([H[opid==k].mean(0) for k in range(3)])
p=plus
for s in [0.5,1,1.5,2,3]:
    with torch.no_grad(): o=net.head_from_resid(H[p]+s*(mu[1]-mu[0])).argmax(-1)
    print(s,'sum',(o==(a[p]+bb[p])%11).float().mean().item(),'diff',(o==(a[p]-bb[p])%11).float().mean().item(),'rev',(o==(bb[p]-a[p])%11).float().mean().item())
# full-h counterfactual: MLP fed hc but direct path from h
with torch.no_grad():
    o=net.W_U(H[p]+net.W_out(torch.relu(Hc[p]@W.T+b))).argmax(-1)
print('MLP gets hc, direct gets h: diff acc',(o==outc[p]).float().mean().item())
# direct-path logit stats vs MLP
with torch.no_grad():
    dl=net.W_U(H); ml=net.W_U(net.W_out(torch.relu(H@W.T+b)))
print('direct logit std',dl.std().item(),'mlp logit std',ml.std().item())
