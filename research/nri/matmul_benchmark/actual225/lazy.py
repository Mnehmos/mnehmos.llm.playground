"""On-demand exact coefficients for Pass Thm 3.1 and tensor powers.
Not a full 9/4 matrix multiplication compiler. No dense arrays allocated.
"""
from fractions import Fraction as Q
from functools import lru_cache
from math import comb
from dataclasses import dataclass


@lru_cache(maxsize=128)
def lagrange(k, n):
    """Coefficients of the k-th interpolation polynomial on nodes 0..n-1."""
    p=[Q(1)]
    for j in range(n):
        if j==k: continue
        q=[Q(0)]*(len(p)+1)
        for i,x in enumerate(p):
            q[i]-=x*j/Q(k-j); q[i+1]+=x/Q(k-j)
        p=q
    return tuple(p)


def checked(i,n):
    if not isinstance(i,int) or not 0<=i<n: raise IndexError(i)
    return i


@dataclass(frozen=True)
class Primal:
    a:int; b:int; c:int; t:int
    def __post_init__(self):
        if min(self.a,self.b,self.c,self.t)<1 or self.c*self.t<self.a+2*self.b-1:
            raise ValueError('invalid packing parameters')
    @property
    def s(self): return self.a+self.b-1
    @property
    def l(self): return self.s+self.t
    @property
    def dims(self): return self.a,(self.c+1)*self.b+self.s,(self.c+1)*self.s+self.b
    @property
    def terms(self): return self.c*self.l
    def coefficient(self,leg,r,i,eps):
        if leg not in (0,1,2): raise ValueError('leg must be 0,1,2')
        checked(r,self.terms); checked(i,self.dims[leg]); eps=Q(eps)
        if not eps: raise ValueError('evaluation must be nonzero')
        j,k=divmod(r,self.l); a,b,c,t,s,l=self.a,self.b,self.c,self.t,self.s,self.l
        if leg==0: return Q(k)**i
        if leg==1:
            if i<(c+1)*b:
                p,v=divmod(i,b)
                return Q(k)**v if p==j else -Q(k)**(t+v) if p==j+1 else Q(0)
            v=i-(c+1)*b-j*t
            return eps*Q(k)**(b+v) if 0<=v<t else Q(0)
        def L(h): return lagrange(k,l)[h] if 0<=h<l else Q(0)
        if i>=(c+1)*s: return L(s+i-(c+1)*s-j*t)/eps
        p,v=divmod(i,s)
        h=v+(p-j)*t
        # Z_p[v] = Z_0[v+pt] - shifted W_j terms for j<p.
        return (L(h) if v+p*t<s else Q(0))-(L(h) if j<p else Q(0))


@dataclass(frozen=True)
class Dual:
    base:Primal
    @property
    def dims(self): a,b,c=self.base.dims; return a,c,b
    @property
    def terms(self): return self.base.terms
    def coefficient(self,leg,r,i,eps):
        # Exchange right and output; reverse left coordinate so full/middle swap.
        if leg==0: return self.base.coefficient(0,r,self.dims[0]-1-checked(i,self.dims[0]),eps)
        if leg not in (1,2): raise ValueError('leg must be 0,1,2')
        return self.base.coefficient(3-leg,r,i,eps)


class ExactPower:
    """Tensor product of primal/dual gadgets, exactified at d+1 nodes.
    The contracted gadget is affine in epsilon, despite Laurent factors.
    Degree <= d for the product; extrapolate its constant at epsilon=0.
    """
    def __init__(self,base,power,alternate=False):
        if not isinstance(power,int) or power<1: raise ValueError('positive power')
        self.base=base; self.power=power; self.alternate=alternate
        a,b,c=base.dims; h=power//2 if alternate else 0
        self.dims=(a**power,b**(power-h)*c**h,c**(power-h)*b**h)
        self.block_terms=base.terms**power
        self.terms=(power+1)*self.block_terms
        self.dual=Dual(base)
    def coefficient(self,leg,r,i):
        if leg not in (0,1,2): raise ValueError('leg must be 0,1,2')
        checked(r,self.terms); checked(i,self.dims[leg]); e,r=divmod(r,self.block_terms); eps=e+1
        v=Q((-1)**e*comb(self.power+1,eps)) if leg==2 else Q(1)
        for p in range(self.power-1,-1,-1):
            g=self.dual if self.alternate and p%2 else self.base
            r,j=divmod(r,g.terms); i,k=divmod(i,g.dims[leg])
            v*=g.coefficient(leg,j,k,eps)
            if not v: break
        return v

    def entry(self,i,j,k,max_terms=100000):
        if self.terms>max_terms: raise ValueError('contraction exceeds explicit budget')
        return sum((self.coefficient(0,r,i)*self.coefficient(1,r,j)*self.coefficient(2,r,k)
                    for r in range(self.terms)),Q(0))

    @property
    def dense_slots(self): return self.terms*sum(self.dims)
