"""Phase V drivers (doc 52 §52.8): D2 decoder birth (first part) and D3 run-time tolerance (second part), as run."""
import sys,json; sys.path.insert(0,'/home/user/mnehmos.llm.playground/research/nri')
import torch
from safetensors.torch import load_file
from nri.model import TinyTyped
import run_swap as RS
from sci import decoder as DC
from concurrent.futures import ProcessPoolExecutor
def job(s):
    torch.set_num_threads(1)
    m=TinyTyped('mass_only',**RS.MODEL_KW); out={}
    steps=list(range(600,2001,20))
    for st in steps:
        m.load_state_dict(load_file(f'/home/user/swap_window/mass_only_s{s}/ckpt/step_{st:05d}/model.safetensors'))
        p=DC.profile_exact(m,n_per=128); out[st]=dict(syn_trained=p['syn_trained'],syn_unseen=p['syn_unseen_all'],syn_far=p['syn_far'],syn_near=p['syn_near'],cls=p['class'])
    for st in (0,300,500,3000,6000):
        m.load_state_dict(load_file(f'/home/user/swap_recon/zoo/mass_only_s{s}/ckpt/step_{st:05d}/model.safetensors'))
        p=DC.profile_exact(m,n_per=128); out[st]=dict(syn_trained=p['syn_trained'],syn_unseen=p['syn_unseen_all'],syn_far=p['syn_far'],syn_near=p['syn_near'],cls=p['class'])
    return s,out
with ProcessPoolExecutor(4) as ex: res=dict(ex.map(job,[307,311,305,304,310]))
json.dump(res,open('/home/user/mnehmos.llm.playground/research/nri/runs/phase5_decoder_birth.json','w'))
W=json.load(open('/home/user/mnehmos.llm.playground/research/nri/runs/swap_recon/window_battery_10step.json'))
for s,o in res.items():
    print(s)
    for st in sorted(o):
        if st%100==0 or st in (0,300,500):
            wb=W[f'mass_only_s{s}'].get(str(st))
            print(f"   {st:5d} syn tr {o[st]['syn_trained']:.2f} syn unseen {o[st]['syn_unseen']:.2f} | real id {wb['acc_id'] if wb else float('nan'):.2f} real oodM {wb['oodM'] if wb else float('nan'):.2f}")
import sys,json; sys.path.insert(0,'/home/user/mnehmos.llm.playground/research/nri')
import torch
from safetensors.torch import load_file
from nri.model import TinyTyped
import run_swap as RS
from sci import rescue as R
from concurrent.futures import ProcessPoolExecutor
def tol(m,k=0.5,r=4,seed=0):
    dec=R.Decomp(m)
    g=torch.Generator().manual_seed(seed)
    scale=float(torch.cat([dec.a,dec.b]).norm(dim=1).mean()); PS=dec.S@dec.S.T
    out={}
    base=R.score(m,lambda h,L:h)
    for orient in ('inS','outS'):
        Z=torch.randn(3,13,r,generator=g,dtype=torch.double); Z=Z-Z.mean(0,keepdim=True)-Z.mean(1,keepdim=True)+Z.mean((0,1),keepdim=True)
        B=torch.linalg.qr(torch.randn(16,16,generator=g,dtype=torch.double))[0][:,:r]
        Br=(PS@B) if orient=='inS' else ((torch.eye(16,dtype=PS.dtype)-PS)@B)
        I=Z@Br.T; I=I/I.norm(dim=-1).mean()
        s=R.score(m,lambda h,L: R.with_block(h,h[:,:16].double()+I[L['mu'],L['m']]*k*scale))
        out[orient]=dict(oodM=s['oodM'],id=s['id'])
    # magnitude-matched isotropic control
    def iso(h,L):
        z=torch.randn(len(h),16,generator=g,dtype=torch.double); z=z/z.norm(dim=1,keepdim=True)*k*scale
        return R.with_block(h,h[:,:16].double()+z)
    s=R.score(m,iso); out['isotropic']=dict(oodM=s['oodM'],id=s['id'])
    out['clean']=dict(oodM=base['oodM'],id=base['id'])
    return out
def job(s):
    torch.set_num_threads(1); m=TinyTyped('mass_only',**RS.MODEL_KW); res={}
    for st in (1200,1400,1600,2000,3000,4000,6000):
        m.load_state_dict(load_file(f'/home/user/swap_recon/zoo/mass_only_s{s}/ckpt/step_{st:05d}/model.safetensors')); res[st]=tol(m)
    return s,res
with ProcessPoolExecutor(4) as ex: res=dict(ex.map(job,[307,311,300,305]))
json.dump(res,open('/home/user/mnehmos.llm.playground/research/nri/runs/phase5/tolerance.json','w'))
for s,r in res.items():
    print(s,' '.join(f"{st}: clean {v['clean']['oodM']:.2f} inS {v['inS']['oodM']:.2f} outS {v['outS']['oodM']:.2f} iso {v['isotropic']['oodM']:.2f} (id {v['inS']['id']:.2f})" for st,v in r.items()))
