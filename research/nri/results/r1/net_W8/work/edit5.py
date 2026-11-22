import sys,os; sys.path.insert(0, os.getcwd())
exec(open('work/edit1.py').read().split("tr=torch")[0])
# train-set success to see if data-limited
w=torch.where(scope,3.0,1.0)[:,None].sqrt()
Dm=torch.linalg.lstsq(X*w,(target-Z)*w).solution
with torch.no_grad(): p=net.head_from_resid(H,W+Dm[:64].T,b+Dm[64]).argmax(1)
print('train(all) success',(p[scope]==y2[scope]).float().mean().item(),'preserve',(p[~scope]==y[~scope]).float().mean().item())
