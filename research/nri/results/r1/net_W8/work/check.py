import sys,os; sys.path.insert(0, os.getcwd())
import json, torch
from arch import load_net
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
doc=json.load(open('nri.json')); L=T.tolist()
B={'__builtins__':{'range':range,'len':len,'min':min,'max':max,'abs':abs,'int':int,'sum':sum}}
V={x['id']:[eval(x['binding'],B)(t) for t in L] for x in doc['variables']}
M={m['id']:m for m in doc['mechanisms']}
with torch.no_grad(): H=net.resid_pre_mlp(T,C); y=net.head_from_resid(H).argmax(1)
mu=H.mean(0)
def fn(mid,vd): return eval(M[mid]['output_fn'],B)(vd)
def rowv(i): return {k:V[k][i] for k in V}
g=torch.Generator().manual_seed(7)
for c in doc['claims']:
    if c['kind']=='behavior':
        acc=sum(fn(c['mechanism'],rowv(i))==y[i].item() for i in range(len(L)))/len(L); print(c['id'],acc)
    elif c['kind'] in('linear_rep','necessity'):
        k=c['variable']; vv=torch.tensor(V[k]); u=vv.unique()
        Mm=torch.stack([H[vv==x].mean(0) for x in u])-mu; B_=torch.linalg.svd(Mm,full_matrices=False)[2][:c['rank']].T; P=B_@B_.T
        if c['kind']=='linear_rep':
            i=torch.randint(0,len(L),(3000,),generator=g); j=torch.randint(0,len(L),(3000,),generator=g)
            with torch.no_grad(): p=net.head_from_resid(H[i]-H[i]@P+H[j]@P).argmax(1)
            ok=0
            for n,(a,b) in enumerate(zip(i.tolist(),j.tolist())):
                vd=rowv(a); vd[k]=V[k][b]; ok+=fn(c['mechanism'],vd)==p[n].item()
            print(c['id'],ok/3000)
        else:
            with torch.no_grad(): p=net.head_from_resid(H-H@P+mu@P).argmax(1)
            print(c['id'],sum(fn(c['mechanism'],rowv(i))==p[i].item() for i in range(len(L)))/len(L))
    elif c['kind']=='edit':
        sc=eval(c['scope'],B); ef=eval(c['edit_fn'],B)
        import edit_program
        exec(open('work/mockhost.py').read().split("import edit_program")[0].split("from arch import load_net")[1].replace("net=load_net(); d=torch.load('inputs.pt')",""))
        dW,db=edit_program.program(Host(),doc)
        with torch.no_grad(): p=net.head_from_resid(H,net.W_in.weight+dW,net.W_in.bias+db).argmax(1)
        s=[sc(rowv(i)) for i in range(len(L))]
        succ=[p[i].item()==ef(rowv(i)) for i in range(len(L)) if s[i]]; pres=[p[i].item()==y[i].item() for i in range(len(L)) if not s[i]]
        print(c['id'],'success',sum(succ)/len(succ),'preserve',sum(pres)/len(pres),dW.shape,db.shape)
