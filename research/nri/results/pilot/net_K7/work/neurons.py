import torch
exec(open('work/probe.py').read().split('for v in V:')[0])
W=net.W_in.weight.detach(); b=net.W_in.bias.detach()
Z=H@W.T+b; A=torch.relu(Z)
op=torch.tensor(V['op'])
fr=torch.stack([(A[op==o]>0).float().mean(0) for o in range(3)])  # 3x256
ma=torch.stack([A[op==o].mean(0) for o in range(3)])
dead=(A.max(0).values<=0)
print('dead',dead.sum().item())
# specificity
share=ma/ma.sum(0).clamp_min(1e-9)
spec=share.argmax(0); sm=share.max(0).values
for o in range(3): print('op',o,'neurons w/ share>0.9:',((spec==o)&(sm>0.9)&~dead).sum().item(),' >0.7:',((spec==o)&(sm>0.7)&~dead).sum().item())
print('mixed (max share<0.7):',((sm<0.7)&~dead).sum().item())
# contribution to logits: importance
Wout=net.W_out.weight.detach(); WU=net.W_U.weight.detach()
imp=(WU@Wout).norm(dim=0)*A.mean(0)
print('top neurons by imp:', [(i, [round(x,2) for x in share[:,i].tolist()]) for i in imp.argsort(descending=True)[:15].tolist()])
# ablation: zero op-specific neurons
for o in range(3):
    S=(spec==o)&(sm>0.7)
    A2=A.clone(); A2[:,S]=0
    o2=net.W_U(H+net.W_out(A2)).argmax(-1)
    print('zero op',o,'neurons -> acc by op',[round((o2==out)[op==k].float().mean().item(),3) for k in range(3)])
