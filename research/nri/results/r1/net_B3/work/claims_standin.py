import sys, json, torch; sys.path.insert(0,'work'); sys.path.insert(0,'.')
torch.set_num_threads(3)
from standin import load, Host, TOK
m=load(); host=Host(m); doc=json.load(open('nri.json'))
V={x['id']:eval(x['binding']) for x in doc['variables']}; f=eval(doc['mechanisms'][0]['output_fn'])
T=TOK[:6000]; L=T.tolist(); vals={k:torch.tensor([fn(t) for t in L]) for k,fn in V.items()}
h=host.site('',T); W,b=m.W_in.weight,m.W_in.bias
head=lambda hh: m.head_from_resid(hh,W,b).argmax(-1)
def proj(var,r):
    v=vals[var]; mu=h.mean(0); Mm=torch.stack([h[v==c].mean(0)-mu for c in v.unique()])
    U=torch.linalg.svd(Mm,full_matrices=False)[2][:r]; return U.T@U
for var,r in (('V_op',2),('V_a',10),('V_b',10)):
    P=proj(var,r); perm=torch.randperm(len(T))
    with torch.no_grad():
        hn=h-h@P+h[perm]@P; p=head(hn)
        exp=torch.tensor([f({**{k:vals[k][i].item() for k in vals},var:vals[var][perm[i]].item()}) for i in range(len(T))])
        ia=(p==exp).float().mean().item()
        ha=h-(h-h.mean(0))@P; pa=head(ha)
        y=torch.tensor([f({k:vals[k][i].item() for k in vals}) for i in range(len(T))])
        print(var,r,'interchange',ia,'ablation acc',(pa==y).float().mean().item())
