import torch,sys; sys.path.insert(0,'.')
import torch.nn.functional as F
from arch import load_net
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
N=len(T); r=torch.arange(N); k=((T>=11)&(T<=13)).float().argmax(1)
A=T[r,k-1]; OP=T[r,k]; B=T[r,k+1]; RES=torch.where(OP==11,A+B,torch.where(OP==12,A-B,A*B))%11
with torch.no_grad(): h=net.resid_pre_mlp(T,C)
tr=torch.arange(14000); te=torch.arange(14000,N)
def meanspan(lab,nc,rank):
    mu=h[tr].mean(0); M=torch.stack([h[tr][lab[tr]==c].mean(0)-mu for c in range(nc)])
    U,S,V=torch.linalg.svd(M,full_matrices=False); return V[:rank].T
def lda(lab,nc,rank):
    H=h[tr]; mu=H.mean(0); Hc=H-mu; M=torch.stack([Hc[lab[tr]==c].mean(0) for c in range(nc)])
    Sw=sum(((Hc[lab[tr]==c]-M[c]).T@(Hc[lab[tr]==c]-M[c])) for c in range(nc))/len(H)+1e-3*torch.eye(64)
    L=torch.linalg.cholesky(Sw); Li=torch.linalg.inv(L); e,v=torch.linalg.eigh(Li@(M.T@M/nc)@Li.T)
    Q,_=torch.linalg.qr(Li.T@v[:,-rank:]); return Q
def acc_probe(Q,lab,nc):
    P=torch.cat([h@Q,torch.ones(N,1)],1); s=torch.linalg.lstsq(P[tr],F.one_hot(lab[tr],nc).float()).solution
    return ((P[te]@s).argmax(1)==lab[te]).float().mean().item()
def necc(Q):
    out=[]
    for mode in ['mean','zero']:
        hm=h-(h-(h[tr].mean(0) if mode=='mean' else 0))@Q@Q.T
        with torch.no_grad(): y=net.head_from_resid(hm).argmax(-1)
        out.append(round((y[te]==RES[te]).float().mean().item(),3))
    return out
for name,lab,nc,ranks in []:
    for rank in ranks:
        for fn in [meanspan,lda]:
            Q=fn(lab,nc,rank); print(name,rank,fn.__name__,'probe',round(acc_probe(Q,lab,nc),3),'ablate',necc(Q),flush=True)
# combined a+b subspace

print('--- RRR / PCA-of-probe')
def rrr(lab,nc,rank):
    H=h[tr]-h[tr].mean(0); Y=F.one_hot(lab[tr],nc).float(); Y=Y-Y.mean(0)
    Bo=torch.linalg.lstsq(H,Y).solution; Fv=H@Bo; _,_,V=torch.linalg.svd(Fv,full_matrices=False)
    Q,_=torch.linalg.qr(Bo@V[:rank].T); return Q
for name,lab,nc,ranks in [('op',OP-11,3,[2]),('a',A,11,[6,10]),('b',B,11,[6,10]),('res',RES,11,[10])]:
    for rank in ranks:
        Q=rrr(lab,nc,rank); print(name,rank,'rrr probe',round(acc_probe(Q,lab,nc),3),'ablate',necc(Q),flush=True)
# full-rank linear probe accuracies
for name,lab,nc in [('op',OP-11,3),('a',A,11),('b',B,11),('res',RES,11),('pos',k-1,4)]:
    print(name,'full probe',round(acc_probe(torch.eye(64),lab,nc),3))
