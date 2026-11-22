import torch,sys; sys.path.insert(0,'.')
import torch.nn.functional as F
from arch import load_net
net=load_net(); d=torch.load('inputs.pt'); T,C=d['tokens'],d['ch2']
W=net.W_in.weight.detach(); b=net.W_in.bias.detach()
with torch.no_grad():
  h=net.resid_pre_mlp(T,C); tc=T.clone(); tc[tc==11]=12; hc=net.resid_pre_mlp(tc,C)
  has=(T==11).any(1)
  # upper bound: direct path h, MLP sees h_cf
  mlp=net.W_out(F.relu(hc@W.T+b)); y=net.W_U(h+mlp).argmax(-1); ycf=net.W_U(hc+net.W_out(F.relu(hc@W.T+b))).argmax(-1)
  print('UB',(y[has]==ycf[has]).float().mean())
  D=(hc-h)[has]; print('delta norm',D.norm(dim=1).mean(), 'h norm', h.norm(dim=1).mean())
  Dm=D-D.mean(0); print('frac var unexplained by mean', (Dm.pow(2).sum()/D.pow(2).sum()).item())
  # UB with constant shift only
  hs=h+has[:,None]*D.mean(0); y2=net.W_U(h+net.W_out(F.relu(hs@W.T+b))).argmax(-1)
  print('const shift',(y2[has]==ycf[has]).float().mean(), (y2[~has]==ycf[~has]).float().mean())
  # linear fit of D from h on has rows
  X=torch.cat([h[has],torch.ones(has.sum(),1)],1); sol=torch.linalg.lstsq(X,D).solution
  R=D-X@sol; print('R2 lin on has', 1-(R.pow(2).sum()/Dm.pow(2).sum()).item())
  hs=h.clone(); hs[has]=h[has]+X@sol; y3=net.W_U(h+net.W_out(F.relu(hs@W.T+b))).argmax(-1)
  print('lin on has only',(y3[has]==ycf[has]).float().mean())
  # dead neurons
  pre=h@W.T+b; print('dead neurons', ((pre>0).float().mean(0)==0).sum().item(), 'rarely', ((pre>0).float().mean(0)<0.01).sum().item())
  # what about op positions / ch2 relation
  S=torch.linalg.svdvals(Dm); print('svals D', (S[:12]/S[0]).numpy().round(3))
