from protected import *

def fuse(h,s):
 mean,sd,W,mu,ss=s;w=h.weight.detach().double()[:,:W.shape[1]]/ss;A=(w@W.T)/sd;b=h.bias.detach().double()-w@mu-A@mean;return A.float(),b.float()

def main():
 rows=[];out=ROOT/'compiled';out.mkdir(exist_ok=False)
 for s,mode,step,p in paths():
  if step!=6000:continue
  name=f'{s}_{mode}';d=ROOT/'protected'/name;td=t.load(d/'train_domain.pt',weights_only=True);st=td['transform'];m=TinyTyped('mass_only',**MODEL_KW);m.load_state_dict(t.load(p,weights_only=True)['model']);m.eval().requires_grad_(False);a=aslat(ALL);raw=views(features(m,ALL),'early');x=[apply(v,c) for v,c in zip(raw,st)];ff=field_factors(a);M,A,S=T.canon(a);C=A*S%13;Y=M*C%13;I=t.arange(len(ALL));D=I[:,None].expand(-1,5);allow=t.zeros(len(ALL),dtype=t.bool);allow[key(aslat(grid(s)))]=True;FAM=allow[key(a)];dd=panels(ff,Y,I[FAM],52595,True);paths0=sorted(d.glob('*32_*.pt'))+[d/'least_squares.pt']+sorted(d.glob('*/h0100.pt'))
  for f in paths0:
   ck=t.load(f,weights_only=True);net=Relative(1);net.load_state_dict(ck['model']);b=Relative(net.refout);h=[net.val,*net.units];dest=[b.val,*b.units];
   with t.no_grad():
    for hh,bb,i in zip(h,dest,[0,1,3,4]):w,c=fuse(hh,st[i]);bb.weight.copy_(w);bb.bias.copy_(c)
   tag=f.stem if f.parent==d else f.parent.name;dst=out/f'{name}_{tag}.pt';t.save(dict(model=b.state_dict(),parent_sha256=sha(p),source_sha256=sha(f),params=sum(x.numel() for x in b.parameters())),dst);a1,c1,y1=infer(net,x,D);a2,c2,y2=infer(b,raw,D);m1,m2,v1=infer(net,x,dd);n1,n2,v2=infer(b,raw,dd);rows.append(dict(backbone=name,model=tag,n=len(D),m_equiv=int((a1==a2).sum()),c_equiv=int((c1==c2).sum()),out_equiv=int((y1==y2).sum()),site_n=len(dd),site_equiv=int((v1==v2).sum()),site_correct=int((v2==target(ff,dd)).sum()),params=sum(x.numel() for x in b.parameters())))
 dumpcsv(ROOT/'compiled_equivalence.csv',rows)
if __name__=='__main__':main()
