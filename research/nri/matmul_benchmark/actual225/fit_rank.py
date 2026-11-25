"""Bounded numerical search for exact rank factors. No floating fit is a proof."""
import os
for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','BLIS_NUM_THREADS'): os.environ[v]='1'
import numpy as np
import json, time


def residual(U,V,W):
    r,n,_=U.shape
    flat=[x.reshape(r,-1) for x in (U,V,W)]
    model2=np.sum((flat[0]@flat[0].T)*(flat[1]@flat[1].T)*(flat[2]@flat[2].T))
    targetdot=np.sum(U*(W@V.transpose(0,2,1)))
    return float(np.sqrt(max(0, n**3+model2-2*targetdot)/n**3))


def update(a,b,rhs):
    r=a.shape[0]; x=a.reshape(r,-1); y=b.reshape(r,-1)
    G=(x@x.T)*(y@y.T)
    # small regularizer for ill-conditioned normal equations
    G.flat[::r+1]+=max(1e-12,1e-8*np.trace(G)/r)
    return np.linalg.solve(G,rhs.reshape(r,-1)).reshape(rhs.shape)


def balance(U,V,W):
    r=U.shape[0]; scales=np.array([np.maximum(np.linalg.norm(x.reshape(r,-1),axis=1),1e-20) for x in (U,V,W)])
    target=np.exp(np.log(scales).mean(axis=0))
    return [x*(target/scales[i])[:,None,None] for i,x in enumerate((U,V,W))]


def fit(n,r,seed,sweeps):
    rng=np.random.default_rng(seed)
    U,V,W=[rng.normal(size=(r,n,n))/n for _ in range(3)]
    history=[]; started=time.perf_counter(); best=float('inf'); best_step=0
    for step in range(sweeps+1):
        e=residual(U,V,W)
        if e<best: best=e; best_step=step
        if step%10==0 or step==sweeps:history.append([step,e])
        if step==sweeps:break
        U=update(V,W,W@V.transpose(0,2,1))
        V=update(U,W,U.transpose(0,2,1)@W)
        W=update(U,V,U@V)
        U,V,W=balance(U,V,W)
    # Independent dense residual for the final iterate.
    target=np.zeros((n*n,)*3)
    for i in range(n):
        for j in range(n):
            for k in range(n):target[i*n+j,j*n+k,i*n+k]=1
    actual=np.einsum('ra,rb,rc->abc',U.reshape(r,-1),V.reshape(r,-1),W.reshape(r,-1),optimize=True)
    check=float(np.linalg.norm(actual-target)/np.linalg.norm(target))
    return {'side':n,'rank':r,'seed':seed,'sweeps':sweeps,'history':history,'best_rel_residual':best,'best_step':best_step,'final_dense_rel_residual':check,'max_abs_coefficient_error':float(np.max(np.abs(actual-target))), 'seconds':time.perf_counter()-started,'qualifying_exact_certificate':False}


if __name__=='__main__':
    out={'numpy':np.__version__,'objective':'Frobenius relative error of matrix multiplication coefficient tensor','method':'regularized alternating least squares; float64; one BLAS thread; no rational recovery unless near exact','runs':[]}
    # Random rank-7 controls demonstrate search behavior, not a guaranteed recovery.
    for n,r,sweeps,seeds in [(2,7,500,[307,311]),(10,223,200,[307,311,305])]:
        for seed in seeds:
            z=fit(n,r,seed,sweeps); out['runs'].append(z)
            print(f'{n=} {r=} {seed=} residual={z["best_rel_residual"]:.8f}',flush=True,file=__import__('sys').stderr)
    print(json.dumps(out,indent=2))
