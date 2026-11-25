"""Exact rational rank certificate verifier and recursive executor, Section 5.
General algebraic coefficients are supported separately in algebraic_search.py.
"""
import argparse
from fractions import Fraction as Q
from itertools import product
import json
from pathlib import Path


class Certificate:
    def __init__(self, block, U, V, W):
        self.u=int(block)
        if self.u<2: raise ValueError('block size must be >=2')
        self.U,self.V,self.W=([list(map(Q,row)) for row in M] for M in (U,V,W))
        self.r=len(self.U); d=self.u**2
        if not (self.r>=d and len(self.V)==self.r and len(self.W)==d and
                all(len(row)==d for row in self.U+self.V) and all(len(row)==self.r for row in self.W)):
            raise ValueError('invalid dimensions')
        self.verified=False

    @classmethod
    def load(cls,p):
        d=json.loads(Path(p).read_text()); return cls(d['block'],d['U'],d['V'],d['W'])

    def verify(self):
        self.verified=False
        u=self.u; d=u*u
        for a,b in product(range(d),repeat=2):
            i,k=divmod(a,u); l,j=divmod(b,u)
            terms=[self.U[t][a]*self.V[t][b] for t in range(self.r)]
            for c in range(d):
                actual=sum(w*t for w,t in zip(self.W[c],terms))
                if actual!=int(k==l and c==i*u+j):
                    raise ValueError(f'bad identity at {a},{b},{c}')
        self.verified=True
        return d**3

    def qualifies(self,epsilon):
        e=Q(9,4)+Q(epsilon)
        if Q(epsilon)<=0: raise ValueError('epsilon must be positive')
        self.verify()
        return self.r**e.denominator < self.u**e.numerator

    def multiply_exact(self,A,B):
        self.verify()
        n=len(A)
        if not n or len(B)!=n or any(len(row)!=n for row in list(A)+list(B)):
            raise ValueError('equal nonempty square matrices required')
        N=1
        while N<n: N*=self.u
        def pad(M): return [[Q(M[i][j]) if i<n and j<n else Q(0) for j in range(N)] for i in range(N)]
        def rec(X,Y):
            s=len(X)
            if s==1: return [[X[0][0]*Y[0][0]]]
            u=self.u; h=s//u
            def blocks(M): return [[row[j*h:(j+1)*h] for row in M[i*h:(i+1)*h]] for i,j in product(range(u),repeat=2)]
            def combine(weights,parts):
                return [[sum((w*P[i][j] for w,P in zip(weights,parts) if w),Q(0)) for j in range(h)] for i in range(h)]
            x,y=blocks(X),blocks(Y)
            ps=[rec(combine(a,x),combine(b,y)) for a,b in zip(self.U,self.V)]
            c=[combine(w,ps) for w in self.W]
            return [[c[(i//h)*u+j//h][i%h][j%h] for j in range(s)] for i in range(s)]
        return [row[:n] for row in rec(pad(A),pad(B))[:n]]


def strassen_control():
    U=[[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]]
    V=[[1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],[0,0,0,1],[1,1,0,0],[0,0,1,1]]
    W=[[1,0,0,1,-1,0,1],[0,0,1,0,1,0,0],[0,1,0,1,0,0,0],[1,-1,1,0,0,1,0]]
    return Certificate(2,U,V,W)


def self_test():
    import random
    c=strassen_control(); checks=c.verify(); rng=random.Random(307)
    for n in (1,2,3,4,5,8):
        a=[[rng.randint(-5,5) for _ in range(n)] for _ in range(n)]
        b=[[rng.randint(-5,5) for _ in range(n)] for _ in range(n)]
        assert c.multiply_exact(a,b)==[[sum(a[i][k]*b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    assert not c.qualifies('1/10')
    c.U[0][0]+=1
    try: c.verify()
    except ValueError: pass
    else: raise AssertionError('corrupt certificate accepted')
    return {'control':'Strassen only','basis_coefficients':checks,'exact_products':6,'corrupt_rejected':True,'qualifies_2_35':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--certificate'); p.add_argument('--epsilon',default='1/10'); args=p.parse_args()
    if args.certificate:
        c=Certificate.load(args.certificate); n=c.verify()
        print(json.dumps({'block':c.u,'rank':c.r,'checks':n,'qualifies':c.qualifies(args.epsilon)}))
    else: print(json.dumps(self_test(),indent=2))
