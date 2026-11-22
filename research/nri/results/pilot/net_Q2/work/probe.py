import torch,sys; sys.path.insert(0,'.')
import torch.nn.functional as F
from arch import load_net
torch.set_num_threads(1)
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
N=len(T); r=torch.arange(N); k=((T>=11)&(T<=13)).float().argmax(1)
A=T[r,k-1]; OP=T[r,k]; B=T[r,k+1]
RES=torch.where(OP==11,A+B,torch.where(OP==12,A-B,A*B))%11
with torch.no_grad(): h=net.resid_pre_mlp(T,C); y=net(T,C).argmax(-1)
print('behavior acc', (y==RES).float().mean().item())
tr=torch.arange(0,14000); te=torch.arange(14000,N)
def lda_basis(H,lab,nc,rank):
    mu=H.mean(0); Hc=H-mu
    M=torch.stack([Hc[lab==c].mean(0) for c in range(nc)])
    Sw=sum(((Hc[lab==c]-M[c]).T@(Hc[lab==c]-M[c])) for c in range(nc))/len(H)+1e-3*torch.eye(H.shape[1])
    Sb=(M.T@M)/nc
    L=torch.linalg.cholesky(Sw); Li=torch.linalg.inv(L)
    e,v=torch.linalg.eigh(Li@Sb@Li.T); V=Li.T@v[:,-rank:]
    Q,_=torch.linalg.qr(V); return Q
def probe(lab,nc,rank):
    Q=lda_basis(h[tr],lab[tr],nc,rank)
    P=h@Q; Xa=torch.cat([P,torch.ones(N,1)],1)
    Y=F.one_hot(lab,nc).float(); s=torch.linalg.lstsq(Xa[tr],Y[tr]).solution
    # also nearest centroid
    cen=torch.stack([P[tr][lab[tr]==c].mean(0) for c in range(nc)])
    nc_acc=((P[te][:,None,:]-cen[None]).pow(2).sum(-1).argmin(1)==lab[te]).float().mean().item()
    lin=((Xa[te]@s).argmax(1)==lab[te]).float().mean().item()
    # necessity: mean-ablate subspace
    hm=h-(h-h[tr].mean(0))@Q@Q.T
    with torch.no_grad(): ya=net.head_from_resid(hm).argmax(-1)
    return round(lin,3),round(nc_acc,3),round((ya[te]==RES[te]).float().mean().item(),3)
for name,lab,nc in [('op',OP-11,3),('a',A,11),('b',B,11),('res',RES,11),('pos',k-1,4)]:
    for rank in ([2] if nc==3 else [2,4,6,10]):
        print(name,rank,probe(lab,nc,rank),flush=True)
