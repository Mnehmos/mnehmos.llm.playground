import sys,os; sys.path.insert(0, os.getcwd())
exec(open('work/edit1.py').read().split("tr=torch")[0])
tr=torch.arange(len(T))<15000; te=~tr
A_t=torch.relu(target)  # desired hidden
for wscope in [1.0,2.0]:
  w=torch.where(scope,wscope,1.0)[:,None].sqrt()
  Xw=X[tr]*w[tr]; pinv=torch.linalg.pinv(Xw)
  tgt=target[tr].clone()
  for it in range(30):
    D=pinv@(tgt*w[tr])
    zc=X[tr]@D
    # where desired hidden is 0 (target<=0) allow any non-positive value
    tgt=torch.where(target[tr]>0,target[tr],torch.minimum(zc,torch.zeros(()) ))
    if it%5==4 or it==0:
      Wn=D[:64].T; bn=D[64]
      with torch.no_grad(): p=net.head_from_resid(H[te],Wn,bn).argmax(1)
      s=scope[te]
      print(wscope,it,'success',round((p[s]==y2[te][s]).float().mean().item(),4),'preserve',round((p[~s]==y[te][~s]).float().mean().item(),4),'rel',round(((Wn-W).norm()/W.norm()).item(),3))
