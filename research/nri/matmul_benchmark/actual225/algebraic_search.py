"""Section 5 existence-to-algorithm bridge. Exact, exhaustive in unbounded mode.
A finite budget does NOT guarantee a certificate; this is not a practical kernel.
Enumerates common number-field coefficients as polynomial residues over Q.
"""
from fractions import Fraction as Q
from itertools import product
import argparse
import json
from pathlib import Path
import time


def qualifies(u,r,epsilon):
    e=Q(9,4)+Q(epsilon)
    if Q(epsilon)<=0: raise ValueError('epsilon must be positive')
    return r**e.denominator < u**e.numerator


class Ring:
    def __init__(self,f):
        self.f=tuple(map(Q,f)); self.d=len(f)-1
        if self.d<1 or self.f[-1]!=1: raise ValueError('monic polynomial required')
        self.zero=(Q(0),)*self.d
        self.one=(Q(1),)+self.zero[1:]
    def add(self,a,b): return tuple(x+y for x,y in zip(a,b))
    def mul(self,a,b):
        c=[Q(0)]*(2*self.d-1)
        for i,x in enumerate(a):
            for j,y in enumerate(b): c[i+j]+=x*y
        for i in range(len(c)-1,self.d-1,-1):
            v=c[i]
            if v:
                for j in range(self.d): c[i-self.d+j]-=v*self.f[j]
        return tuple(c[:self.d])
    def scalar(self,x): return (Q(x),)+self.zero[1:]


def verify(u,r,R,U,V,W):
    d=u*u
    if u<2 or r<d or len(U)!=r or len(V)!=r or len(W)!=d or any(len(x)!=d for x in U+V) or any(len(x)!=r for x in W):
        return False
    if any(len(x)!=R.d for M in (U,V,W) for row in M for x in row): return False
    for a,b in product(range(d),repeat=2):
        i,k=divmod(a,u); l,j=divmod(b,u)
        t=[R.mul(U[s][a],V[s][b]) for s in range(r)]
        for c in range(d):
            v=R.zero
            for w,p in zip(W[c],t): v=R.add(v,R.mul(w,p))
            if v!=(R.one if k==l and c==i*u+j else R.zero): return False
    return True


def multiply_exact(u,r,R,U,V,W,A,B):
    # Coefficients live in a commutative quotient algebra. Rational inputs embed
    # into it; the exact multiplication identities imply rational outputs.
    if not verify(u,r,R,U,V,W): raise ValueError('invalid certificate')
    n=len(A)
    if n<1 or len(B)!=n or any(len(row)!=n for row in list(A)+list(B)):
        raise ValueError('equal nonempty square matrices required')
    N=1
    while N<n: N*=u
    def pad(M):
        return [[R.scalar(M[i][j]) if i<n and j<n else R.zero for j in range(N)] for i in range(N)]
    def rec(X,Y):
        s=len(X)
        if s==1: return [[R.mul(X[0][0],Y[0][0])]]
        h=s//u
        def blocks(M):
            return [[row[j*h:(j+1)*h] for row in M[i*h:(i+1)*h]] for i,j in product(range(u),repeat=2)]
        def combine(weights,parts):
            out=[[R.zero for j in range(h)] for i in range(h)]
            for w,P in zip(weights,parts):
                if w==R.zero: continue
                for i,j in product(range(h),repeat=2):
                    out[i][j]=R.add(out[i][j],R.mul(w,P[i][j]))
            return out
        X,Y=blocks(X),blocks(Y)
        ps=[rec(combine(a,X),combine(b,Y)) for a,b in zip(U,V)]
        c=[combine(w,ps) for w in W]
        return [[c[(i//h)*u+j//h][i%h][j%h] for j in range(s)] for i in range(s)]
    C=[row[:n] for row in rec(pad(A),pad(B))[:n]]
    if any(x[1:]!=R.zero[1:] for row in C for x in row):
        raise ArithmeticError('unexpected nonrational output')
    return [[x[0] for x in row] for row in C]


def load_found(path):
    d=json.loads(Path(path).read_text()); d=d.get('search',d)
    if d.get('status')!='FOUND_EXACT_CERTIFICATE': raise ValueError('no certificate')
    R=Ring(d['f']); u,r=int(d['block']),int(d['rank'])
    U,V,W=([[tuple(map(Q,x)) for x in row] for row in d[k]] for k in ('U','V','W'))
    if any(len(x)!=R.d for M in (U,V,W) for row in M for x in row): raise ValueError('invalid coefficient length')
    if not qualifies(u,r,d['epsilon']) or not verify(u,r,R,U,V,W): raise ValueError('invalid certificate')
    return u,r,R,U,V,W


def search(epsilon,max_height,max_candidates,max_seconds):
    # Height h covers u, degree, coefficient numerators and denominators <= h.
    # Repeated candidates are allowed. Every finite algebraic certificate is covered
    # at some finite height (if the theorem holds). r is bounded by the exact threshold.
    started=time.monotonic(); checked=0; h=1
    def expired():
        return ((max_candidates is not None and checked>=max_candidates) or
                (max_seconds is not None and time.monotonic()-started>=max_seconds))
    while max_height is None or h<=max_height:
        rationals=sorted({Q(a,b) for a in range(-h,h+1) for b in range(1,h+1)},key=lambda q:(abs(q),q))
        for u in range(2,h+1):
            d=u*u
            for r in range(d,u**3+1):
                if not qualifies(u,r,epsilon): continue
                for degree in range(1,h+1):
                    for f in product(rationals,repeat=degree):
                        if expired(): return {'status':'NOT_FOUND_WITHIN_BUDGET','checked':checked,'height':h}
                        R=Ring(f+(Q(1),))
                        for flat in product(rationals,repeat=3*d*r*degree):
                            if expired(): return {'status':'NOT_FOUND_WITHIN_BUDGET','checked':checked,'height':h}
                            co=[flat[i:i+degree] for i in range(0,len(flat),degree)]
                            U=[co[i*d:(i+1)*d] for i in range(r)]
                            V=[co[d*r+i*d:d*r+(i+1)*d] for i in range(r)]
                            W=[co[2*d*r+i*r:2*d*r+(i+1)*r] for i in range(d)]
                            checked+=1
                            if verify(u,r,R,U,V,W):
                                return {'status':'FOUND_EXACT_CERTIFICATE','checked':checked,'height':h,'block':u,'rank':r,
                                        'epsilon':str(Q(epsilon)),'f':[str(x) for x in R.f],
                                        'U':U,'V':V,'W':W}
        h+=1
        if expired(): break
    return {'status':'NOT_FOUND_WITHIN_BUDGET','checked':checked,'height':h}


def self_test():
    from certificate import strassen_control
    c=strassen_control(); c.verify()
    R=Ring([-2,0,1]); t=(Q(0),Q(1))
    assert R.mul(t,t)==R.scalar(2)
    U,V,W=([[R.scalar(x) for x in row] for row in m] for m in (c.U,c.V,c.W))
    U[0]=[R.mul(t,x) for x in U[0]]
    V[0]=[R.mul((Q(0),Q(1,2)),x) for x in V[0]]
    assert verify(2,7,R,U,V,W)
    A=[[1,2,3],[0,2,1],[3,0,1]]; B=[[2,0,1],[1,2,0],[1,0,3]]
    assert multiply_exact(2,7,R,U,V,W,A,B)==[[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    U[0][0]=R.add(U[0][0],R.one)
    assert not verify(2,7,R,U,V,W)
    return {'algebraic_control_pass':True,'recursive_algebraic_product_pass':True,'corruption_rejected':True,'control_not_2_25':not qualifies(2,7,'1/10')}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--epsilon',default='1/10'); p.add_argument('--max-height',type=int,default=2)
    p.add_argument('--max-candidates',type=int,default=100); p.add_argument('--max-seconds',type=float,default=5)
    p.add_argument('--unbounded',action='store_true'); p.add_argument('--output',type=Path); args=p.parse_args()
    if Q(args.epsilon)<=0: p.error('epsilon must be positive')
    result={'self_test':self_test(),'search':search(args.epsilon,None if args.unbounded else args.max_height,
        None if args.unbounded else args.max_candidates,None if args.unbounded else args.max_seconds)}
    s=json.dumps(result,default=str,indent=2)
    if args.output:
        with args.output.open('x') as f: f.write(s+'\n')
    print(s)
