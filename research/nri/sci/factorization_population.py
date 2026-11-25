import sys,json; sys.path.insert(0,'/home/user/mnehmos.llm.playground/research/nri')
import torch
from sci import window_analysis as W
from nri import dimtask as T
from nri.model import TinyTyped
import run_swap as RS
from safetensors.torch import load_file
from concurrent.futures import ProcessPoolExecutor
P=13
@torch.no_grad()
def r2_add(m, var, n=16384):
    L=W.lat(n,7)
    b,x1,h,_,_,_=W.states(m,L); X=h[:,:16].double()
    vk,uk,S=('m','mu',T.MASS_S) if var=='M' else ('a','au',T.ACC_S)
    keys=sorted(set(zip(L[uk].tolist(),L[vk].tolist())))
    C=torch.stack([X[(L[uk]==u)&(L[vk]==v)].mean(0) for u,v in keys]); gm=C.mean(0); tot=float(((C-gm)**2).sum())
    D=torch.zeros(len(keys),16,dtype=torch.double)
    for i,(u,v) in enumerate(keys): D[i,v]=1; D[i,13+u]=1
    co=torch.linalg.lstsq(D,C).solution
    return 1-float(((C-D@co)**2).sum())/tot
def job(name):
    torch.set_num_threads(1)
    g=name.rsplit('_s',1)[0]; var='M' if g=='mass_only' else 'A'
    m=TinyTyped(g,**RS.MODEL_KW); out={}
    for st in list(range(0,3001,100))+[4000,5000,6000]:
        m.load_state_dict(load_file(f'/home/user/swap_recon/zoo/{name}/ckpt/step_{st:05d}/model.safetensors'))
        out[st]=r2_add(m,var)
    return name,out
names=[f'{g}_s{s}' for g in ('mass_only','acc_only') for s in range(300,312)]
with ProcessPoolExecutor(4) as ex: res=dict(ex.map(job,names))
json.dump(res,open('/home/user/mnehmos.llm.playground/research/nri/runs/swap_recon/factorization_trajectories.json','w'))
