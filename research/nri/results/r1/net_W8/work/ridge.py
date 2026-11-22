exec(open('work/mockhost.py').read().split("dW,db=edit_program")[0])
g=torch.Generator().manual_seed(123); T,C=Host().sampler(5000,g)
T2=T.clone(); T2[T2==11]=12
W=net.W_in.weight.detach(); b=net.W_in.bias.detach()
with torch.no_grad(): H=net.resid_pre_mlp(T,C); y=net(T,C).argmax(1); y2=net(T2,C).argmax(1)
s=(T==11).any(1)
for r in [1e-3,1e2,1e3,1e4,3e4,1e5]:
    dW,db=edit_program.program(Host(),doc,ridge=r)
    with torch.no_grad(): p=net.head_from_resid(H,W+dW,b+db).argmax(1)
    print(r,'success',round((p[s]==y2[s]).float().mean().item(),4),'preserve',round((p[~s]==y[~s]).float().mean().item(),4),'relnorm',round((dW.norm()/W.norm()).item(),3))
