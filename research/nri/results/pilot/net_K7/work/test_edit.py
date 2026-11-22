import torch, sys, json, time
sys.path.insert(0,'.'); sys.path.insert(0,'work')
from mockhost import Host
import edit_program
host=Host(); net=host.model
t0=time.time(); dW,db=edit_program.program(host, json.load(open('nri.json')) if False else {})
print('shapes',tuple(dW.shape),tuple(db.shape),'time',round(time.time()-t0,1))
d=torch.load('inputs.pt'); T,C=d['tokens'][14000:],d['ch2'][14000:]
T2=T.clone(); T2[T2==11]=12
with torch.no_grad():
    H=net.resid_pre_mlp(T,C); out=net(T,C).argmax(-1); outc=net(T2,C).argmax(-1)
    o=net.head_from_resid(H,net.W_in.weight+dW,net.W_in.bias+db).argmax(-1)
p=(T==11).any(1)
print('success',(o[p]==outc[p]).float().mean().item(),'preserve',(o[~p]==out[~p]).float().mean().item())
for k in (12,13): m=(T==k).any(1); print('preserve op',k,(o[m]==out[m]).float().mean().item())
