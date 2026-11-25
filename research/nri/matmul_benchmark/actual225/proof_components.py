"""Finite identities from OpenAI Oct 2 upper-bound paper; not a 9/4 kernel."""
from fractions import Fraction as Q
from itertools import product
import json
import sympy as S


def separation(M):
    # Proposition 3.1: evaluate the Fourier sum exactly in Q[z]/Phi_(5M).
    z=S.Symbol('z'); L=5*M
    phi=S.Poly(S.cyclotomic_poly(L,z),z,domain=S.QQ)
    residues={}
    for q in range(-3*(M-1),3*(M-1)+1):
        polynomial=S.Poly(sum(z**((r*q)%L) for r in range(L)),z,domain=S.QQ)
        residue=polynomial.rem(phi)
        assert residue.as_expr()==(L if q==0 else 0)
        residues[q]=q==0
    kept=set(); checked=0
    for h,g,u,v in product(range(1,M+1),repeat=4):
        q=u-v+2*(g-h)
        if residues[q]:
            w=g*g+h*u-h*h-h*v
            assert w==(g-h)**2
            if w==0: kept.add((h,g,u,v))
        checked+=1
    assert kept=={(h,h,u,u) for h in range(1,M+1) for u in range(1,M+1)}
    return checked


def basis(e):
    # coordinates (v exponent, second variable s=0 or w=1).
    d=2*(e+1); B=S.zeros(d)
    for i in range(e+1): B[2*i,i]=1
    B[2*e+1,e+1]=1
    for j in range(e):
        B[2*j+1,e+2+j]=1
        B[2*(j+1),e+2+j]=-1
    return B


def filtration(a,b):
    # Discrete concavity lemma: all first-input monomials preserve kernel.
    e=b-1; f=a+b-2; Bi=basis(e); Bo=basis(f).inv()
    for x in range(a):
        M=S.zeros(2*(f+1),2*(e+1))
        for i in range(e+1):
            for t in range(2): M[2*(i+x)+t,2*i+t]=1
        R=Bo*M*Bi
        assert R[:f+2,e+2:]==S.zeros(f+2,e)
        for i in range(e+2):
            assert R[:f+2,i]==S.eye(f+2)[:,i+x]
        for i in range(e):
            assert R[f+2:,e+2+i]==S.eye(f)[:,i+x]
    return a


def tripling(a,h):
    B=3*h+a-1
    def ys(y): return 0 if y<h else (1 if y<2*h+a-1 else 2)
    def zs(z): return 0 if z<h+a-1 else (1 if z<2*h+a-1 else 2)
    kept=set()
    for x,y in product(range(a),range(B)):
        z=x+y; w=int(ys(y)==1)-int(zs(z)==1)
        assert w>=0
        if w==0: kept.add((x,y,z))
    expected=set()
    for x,r in product(range(a),range(h)):
        expected.add((x,r,x+r))
        expected.add((x,2*h+a-1+r,2*h+a-1+r+x))
        expected.add((a-1-x,h+x+r,h+a-1+r))
    assert kept==expected
    return a*B


def growth(limit=128):
    H=Q(1)
    for a in range(1,limit+1):
        if a>1: H*=1+Q(1,3*(a-1))
        assert H**3>=a
        D=Q(3*a-1,2)*H
        assert D**3>=a**4
    return limit


if __name__=='__main__':
    out={'separation_cases':{m:separation(m) for m in range(1,7)},
         'filtration_slices':sum(filtration(a,b) for a in range(1,6) for b in range(2,7)),
         'tripling_terms':sum(tripling(a,h) for a in range(1,7) for h in range(1,7)),
         'growth_checks':growth(),
         'actual_9_4_kernel':'BLOCKED: finite rank certificate not supplied'}
    print(json.dumps(out,indent=2))
