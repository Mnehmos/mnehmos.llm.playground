"""Read-only retrospective replay of archived converter logits."""
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
import sys
import torch
import study_model as e

torch.set_num_threads(1)

def decomposition(prog,res,y):
    keep=torch.arange(prog.shape[-1])[None,:]!=y[:,None]
    p=(prog.gather(1,y[:,None])-prog)[keep].reshape(len(y),-1)
    c=(res-res.gather(1,y[:,None]))[keep].reshape(len(y),-1)
    m=p-c
    lower=torch.where(c<0,p/c,-torch.inf).amax(1)
    upper=torch.where(c>0,p/c,torch.inf).amin(1)
    possible=((c!=0)|(p>0)).all(1)&(upper>lower)
    return p,c,m,lower,upper,possible

@torch.no_grad()
def score(s, base, donors, name):
    if donors is None:i,j,y=base,base,e.Y[base]
    else:
        (i,j,y,_),_=e.split_swaps(base,donors)
    raw=s.trunk(e.cv_tok[i])[:,:13]
    p,r=s.conversion(i,j,raw=raw)
    pm,cm,margin,lo,hi,valid=decomposition(p,r,y)
    lower=lo.numpy();upper=hi.numpy()
    out=dict(panel=name,n=len(y),bases=len(i.unique()),program_min=float(pm.min()),
      program_median=float(pm.amin(1).median()),competing_median=float(cm.amax(1).median()),
      residual_abs_max=float(r.abs().max()),effective_p10=float(margin.amin(1).quantile(.1)),
      effective_median=float(margin.amin(1).median()),
      effective_p90=float(margin.amin(1).quantile(.9)),authority=e.rate(p+r,y),
      unique=float((margin.amin(1)>0).double().mean()),
      mean_allowed_alpha=float(torch.where(valid,hi.clamp(0,1),0.).mean()))
    for alpha in (0.,.125,.5,1.):
        proposed=(valid&(alpha>lo)&(alpha<hi))
        actual=(p+alpha*r).argmax(1)==y
        ties=(pm-alpha*cm).amin(1).abs()<1e-5
        assert torch.equal(proposed[~ties],actual[~ties])
        out[f'actual_{alpha}']=float(actual.double().mean())
    data=dict(program=pm.numpy(),competing=cm.numpy(),lower=lower,upper=upper,valid=valid.numpy())
    return out,data

def scan(folder,dest):
    import numpy as np
    dest.mkdir(parents=True,exist_ok=True)
    rows=[]; raw={}
    for archive in sorted(folder.glob('Paper_Three_Confirmation_83*.zip'))+sorted(folder.glob('Experiment_31_*.zip')):
      with ZipFile(archive) as z:
        names=z.namelist()
        for cp in [x for x in names if x.endswith('.pt') and not x.startswith(('prehost/','parity/'))]:
            parts=Path(cp).parts
            study='e30' if len(parts)==2 else parts[0]
            if study not in ('e30','repair','host'):continue
            tag=parts[-2]; seed,arm,*_=tag.split('-'); t=int(Path(cp).stem)
            path0=str(Path(cp).with_suffix('.json'))
            card=json.loads(z.read(path0))
            data=z.read(cp)
            assert hashlib.sha256(data).hexdigest()==card['hash']
            ck=torch.load(io.BytesIO(data),weights_only=True)
            s=e.Student(int(seed),arm);s.load_state_dict(ck['model'])
            conv=e.conv_ids
            if study=='host':
                held=(e.u==1)&((e.v>=8)|(e.d==2))
                conv=torch.where(~held)[0];held=torch.where(held)[0]
                sh=held[torch.randperm(len(held),generator=torch.Generator().manual_seed(830079))]
                base=sh[:len(sh)//2].sort().values
            elif study=='repair':
                with ZipFile(folder/'Paper_Three_Confirmation_Data.zip') as dz:
                    pri=json.loads(dz.read('released_manifest.json'))
                base=torch.tensor(sorted(sum(pri['splits']['final'].values(),[])))
            else:
                pub=json.loads(z.read('public.json'))
                base=torch.tensor(sorted(sum(pub['visible'].values(),[])))
            for label,b,d in [('ordinary_train',conv,None),('ordinary_held',base,None),('swap_train',conv,conv),('swap_held',base,conv)]:
                row,a=score(s,b,d,label)
                rows.append(dict(study=study,run=tag,step=t,**row))
                if label=='swap_held':
                    key=f'{study}/{tag}/{t}'
                    raw[key+'_pm']=a['program'].astype('float32')
                    raw[key+'_cm']=a['competing'].astype('float32')
                    raw[key+'_alpha_max']=a['upper'].astype('float32')
            if study=='e30' and t==600 and arm in 'ABDF':
                saved=json.loads(z.read('sealed_scores.json'))[tag]['repair_authority']
                pub=json.loads(z.read('released_manifest.json'))
                bases=torch.tensor(sorted(sum(pub['splits']['final'].values(),[])))
                (i,j,y,_),_=e.split_swaps(bases,conv)
                with torch.no_grad():
                    p,r=s.conversion(i,j)
                    for a in (0.,.125,.5):
                       assert abs(float(e.rate(p+a*r,y))-saved[str(a)])<1e-8
    with (dest/'margins.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(dest/'held_pair_logits.npz',**raw)
    print(len(rows),'panel records with algebraic checks on every non-tied counterfactual')

if __name__=='__main__':
    scan(Path(sys.argv[1]),Path(sys.argv[2]))
