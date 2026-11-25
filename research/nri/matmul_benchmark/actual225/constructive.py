"""Pass (arXiv:2610.28627v2), Thm 3.1: exact coefficient extraction.
Not a qualifying square-matrix multiplication certificate.
"""
from fractions import Fraction as F
from math import comb, log, log1p, ceil
from itertools import product
import json


def convolution(a,b):
    out=[0]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b): out[i+j]+=x*y
    return out


def primal(a,b,c,t,A,B,eps):
    """B contains c+1 length-b polynomials, then shifted middle input of length s."""
    s=a+b-1
    assert c*t>=s+b and eps!=0
    P,Q=B[:-1],B[-1]
    assert len(A)==a and len(P)==c+1 and all(len(p)==b for p in P) and len(Q)==s
    W=[]
    for j in range(c):
        E=[0]*(b+t)
        for i in range(b):
            E[i]+=P[j][i]; E[t+i]-=P[j+1][i]
        for i in range(t):
            if j*t+i<s: E[b+i]+=eps*Q[j*t+i]
        W.append(convolution(A,E))
    total=[0]*(c*t+s)
    for j,source in enumerate(W):
        for i,v in enumerate(source): total[j*t+i]+=v
    Z=[total[:s]]
    for source in W:
        pad=Z[-1]+[0]*t
        Z.append([pad[i+t]-source[i+t] for i in range(s)])
    return [x for p in Z for x in p]+[F(total[s+i],eps) for i in range(b)]


def direct(a,b,A,B):
    return [x for p in B[:-1] for x in convolution(A,p)]+convolution(A,B[-1])[a-1:a-1+b]


def verify_primal(a,b,c,t):
    s=a+b-1; shapes=[b]*(c+1)+[s]; count=0
    for i in range(a):
        A=[int(j==i) for j in range(a)]
        for k,n in enumerate(shapes):
            for j in range(n):
                B=[[int(k==l and j==q) for q in range(m)] for l,m in enumerate(shapes)]
                y1=primal(a,b,c,t,A,B,1); y2=primal(a,b,c,t,A,B,2)
                assert [2*x-y for x,y in zip(y1,y2)]==direct(a,b,A,B)
                count+=1
    return {'a':a,'b':b,'c':c,'t':t,'basis_pairs':count,'target_effort':(c+2)*s,
            'source_effort_before_exact_interpolation':c*(s+t),
            'source_effort_after_two_evaluations':2*c*(s+t)}


def exact_upper_rank_limit(n,tau):
    tau=F(tau); lo,hi=0,n**3+1
    while hi-lo>1:
        r=(lo+hi)//2
        if r**tau.denominator<n**tau.numerator:lo=r
        else:hi=r
    return lo


def lower(n): return max(2*n*n-1,ceil(F(5,2)*n*n-3*n))


def plan(tau):
    """Numerical parameter scan for Pass Thm 4.3. Counts are log estimates,
    not a certificate or a verified full algorithm. Bound L_k by C*rho^k.
    Use floor(w/u) >= w/(2u) for w/u >=2, and lower bound w by formula.
    """
    tau=float(tau)
    c0=next(c for c in range(2,100000) if 1.5*log1p(3/(c-1))/log1p(2/c)<tau)
    best=None
    for c in range(c0+1,8*c0,max(1,c0//20)):
        lrho=log1p(3/(c-1)); lrate=log1p(2/c)
        for j in range(1,33):
            d=2**j
            lw0=2*d*log(c+2)-log(d+1)
            lratio=2*d*lrate-log(d+1)
            if lratio<log(2):continue
            # conservative logs: ceil(x)<=2x, floor(y)>=y/2
            ll=2*(lratio-log(2))
            gap=tau*ll-log(64)-6*d*lrho
            if gap<=0:continue
            # log upper cost - tau log lower side. use same w with coefficient 9-2tau>0.
            lw1=lw0+log(2)
            C=1+(c-1)/3
            k=max(1,ceil(((9-2*tau)*lw1+6*d*log(2*C)+1)/gap))
            lv=2*lw0+k*ll
            item={'target':tau,'c':c,'d':d,'rounds':k,'log10_side_lower_estimate':lv/log(10),
                  'log10_approx_rank_upper_estimate':(9*lw1+6*d*log(2*C)+k*(log(64)+6*d*lrho))/log(10),
                  'log_margin_estimate':k*gap-((9-2*tau)*lw1+6*d*log(2*C)),
                  'note':'Before Bini exact conversion; numeric planning only, not a finite certificate.'}
            if best is None or item['log10_side_lower_estimate']<best['log10_side_lower_estimate']:best=item
            break
    return best


if __name__=='__main__':
    out={'primal_checks':[verify_primal(*x) for x in [(1,1,2,1),(2,2,5,1),(4,4,12,1),(9,9,27,1)]],
         'excluded_prefix':{tau:next(n for n in range(2,1000) if lower(n)<=exact_upper_rank_limit(n,tau))-1 for tau in ['2.35','2.3','2.26','2.251']},
         'rank_screen_2_35':[{'side':n,'rank_max':exact_upper_rank_limit(n,'2.35'),'known_lower':lower(n)} for n in [2,3,4,8,16,32,64]],
         'parameter_plans':[plan(tau) for tau in [2.35,2.3,2.26,2.251]]}
    print(json.dumps(out,indent=2))
