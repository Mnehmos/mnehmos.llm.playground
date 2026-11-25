"""MM-01 finite feasibility. No candidate here implements exponent 9/4."""
import argparse
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path

import torch

SHAPES = [(1,32,32,32),(1,64,64,64),(1,256,256,256),(1,1024,1024,1024),
          (1,4096,64,192),(1,4096,64,64),(1,512,64,256),(1,512,256,64),
          (1,512,64,13),(2048,8,16,8),(2048,8,8,16)]
CAP = 2**20


def strassen(a, b):
    m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
    assert not (m%2 or k%2 or n%2)
    a11,a12,a21,a22 = a[..., :m//2,:k//2],a[..., :m//2,k//2:],a[..., m//2:,:k//2],a[..., m//2:,k//2:]
    b11,b12,b21,b22 = b[..., :k//2,:n//2],b[..., :k//2,n//2:],b[..., k//2:,:n//2],b[..., k//2:,n//2:]
    p1=(a11+a22)@(b11+b22)
    p2=(a21+a22)@b11
    p3=a11@(b12-b22)
    p4=a22@(b21-b11)
    p5=(a11+a12)@b22
    p6=(a21-a11)@(b11+b12)
    p7=(a12-a22)@(b21+b22)
    return torch.cat((torch.cat((p1+p4-p5+p7,p3+p5),-1),
                      torch.cat((p2+p4,p1-p2+p3+p6),-1)),-2)


def packed_fft(a,b):
    batch,m,k=a.shape
    n=b.shape[-1]
    length=m*k*n
    assert batch*length<=CAP
    i,t,j=[torch.arange(x,device=a.device) for x in (m,k,n)]
    ia=(i[:,None]*n*k+t[None,:]).flatten()
    ib=(j[None,:]*k+k-1-t[:,None]).flatten()
    out=(torch.arange(m*n,device=a.device)*k+k-1)
    aa=a.new_zeros((batch,length)).index_copy(-1,ia,a.reshape(batch,-1))
    bb=b.new_zeros((batch,length)).index_copy(-1,ib,b.reshape(batch,-1))
    c=torch.fft.irfft(torch.fft.rfft(aa)*torch.fft.rfft(bb),n=length)
    return c.index_select(-1,out).reshape(batch,m,n)


def evaluate(fn,a,b,g,backward):
    if backward:
        x=a.detach().requires_grad_()
        y=b.detach().requires_grad_()
        c=fn(x,y)
        return (c,*torch.autograd.grad(c,(x,y),g))
    with torch.no_grad():
        return (fn(a,b),)


def err(x,y):
    return float((x.double()-y).norm()/y.norm().clamp_min(1e-30))


def run(device):
    torch.set_num_threads(1)
    if device=="cuda":
        torch.backends.cuda.matmul.allow_tf32=False
    gen=torch.Generator(device=device).manual_seed(307)
    rng=random.Random(307)
    sync=torch.cuda.synchronize if device=="cuda" else lambda:None
    results=[]
    for batch,m,k,n in SHAPES:
        a=torch.randn(batch,m,k,generator=gen,device=device)
        b=torch.randn(batch,k,n,generator=gen,device=device)
        g=torch.randn(batch,m,n,generator=gen,device=device)
        ref=evaluate(torch.matmul,a.double(),b.double(),g.double(),True)
        methods={"native":torch.matmul}
        skipped={}
        if m%2 or k%2 or n%2: skipped["strassen"]="odd dimension"
        else: methods["strassen"]=strassen
        if batch*m*k*n>CAP: skipped["packed_fft"]="packing cap"
        else: methods["packed_fft"]=packed_fft
        row={"shape":[batch,m,k,n],"skipped":skipped,"methods":{}}
        for name,fn in methods.items():
            vals=evaluate(fn,a,b,g,True)
            errors=[err(x,y) for x,y in zip(vals,ref)]
            row["methods"][name]={"relative_l2":errors,"pass":all(math.isfinite(e) and e<=1e-5 for e in errors)}
        for backward in (False,True):
            key="forward_backward_ms" if backward else "forward_ms"
            for name,fn in methods.items():
                for _ in range(3): evaluate(fn,a,b,g,backward)
                row["methods"][name][key]=[]
            sync()
            for _ in range(7):
                order=list(methods)
                rng.shuffle(order)
                for name in order:
                    sync()
                    start=time.perf_counter()
                    evaluate(methods[name],a,b,g,backward)
                    sync()
                    row["methods"][name][key].append((time.perf_counter()-start)*1000)
            base=statistics.median(row["methods"]["native"][key])
            for item in row["methods"].values():
                med=statistics.median(item[key])
                item[key+"_median"]=med
                item[key+"_speedup"]=base/med
        results.append(row)
    cpu=platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu=next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),cpu)
    return {"device":device,"cpu":cpu,"gpu":torch.cuda.get_device_name() if device=="cuda" else None,
            "platform":platform.platform(),"torch":str(torch.__version__),"threads":torch.get_num_threads(),
            "seed":307,"dtype":"float32","results":results}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=run(args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
