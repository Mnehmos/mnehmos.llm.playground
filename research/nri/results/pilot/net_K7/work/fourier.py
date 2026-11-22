import torch
exec(open('work/match.py').read().split('P=CM[0]')[0])
torch.save(CM,'work/CM.pt')
F=torch.fft.fft2(CM.to(torch.complex64),dim=(1,2))  # 3,11,11,64
E=(F.abs()**2).sum(-1)  # 3,11,11
E[:,0,0]=0
for o in range(3):
    e=E[o]/E[o].sum()
    # fold conjugates: (ka,kb) ~ (-ka,-kb)
    items=[]
    for ka in range(11):
        for kb in range(11):
            if (ka,kb)<=((-ka)%11,(-kb)%11):
                v=e[ka,kb]+(e[(-ka)%11,(-kb)%11] if (ka,kb)!=((-ka)%11,(-kb)%11) else 0)
                items.append((v.item(),ka,kb))
    items.sort(reverse=True)
    print('op',o,[(ka,kb,round(v,3)) for v,ka,kb in items[:14]])
    # category: pure a, pure b, sum-diag (k,k), diff-diag (k,-k), other
    cat={'a':0,'b':0,'k,k':0,'k,-k':0,'other':0}
    for v,ka,kb in items:
        if kb==0: cat['a']+=v
        elif ka==0: cat['b']+=v
        elif ka==kb: cat['k,k']+=v
        elif (ka+kb)%11==0: cat['k,-k']+=v
        else: cat['other']+=v
    print('   ',{k:round(v,3) for k,v in cat.items()})
