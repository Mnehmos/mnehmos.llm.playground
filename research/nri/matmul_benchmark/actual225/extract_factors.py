"""Explicit rational factors for Pass's elementary same-left packing.
The factors here compute a polynomial map, not square matrix multiplication.
"""
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
from constructive import direct, convolution


def lagrange_coeff(k, n):
    p=[F(1)]
    for j in range(n):
        if j!=k: p=[x/F(k-j) for x in convolution(p,[-j,1])]
    return p


def decode(W,a,b,c,t,eps):
    s=a+b-1; total=[F(0)]*(c*t+s)
    for j,source in enumerate(W):
        for i,v in enumerate(source): total[j*t+i]+=v
    Z=[total[:s]]
    for source in W:
        pad=Z[-1]+[F(0)]*t
        Z.append([pad[i+t]-source[i+t] for i in range(s)])
    return [x for p in Z for x in p]+[total[s+i]/eps for i in range(b)]


def factors(a,b,c,t):
    s=a+b-1; assert c*t>=s+b
    D=(c+1)*b+s; O=(c+1)*s+b; l=s+t
    U=[]; V=[]; WT=[]
    for eps,scale in [(F(1),2),(F(2),-1)]:
        for j,k in product(range(c),range(l)):
            u=[F(k)**i for i in range(a)]
            v=[F(0)]*D
            for i in range(b):
                v[j*b+i]+=F(k)**i; v[(j+1)*b+i]-=F(k)**(t+i)
            for i in range(t):
                if j*t+i<s:v[(c+1)*b+j*t+i]+=eps*F(k)**(b+i)
            W=[[F(0)]*l for _ in range(c)];W[j]=lagrange_coeff(k,l)
            w=[scale*x for x in decode(W,a,b,c,t,eps)]
            U.append(u);V.append(v);WT.append(w)
    W=list(map(list,zip(*WT)))
    checked=0
    for i,j in product(range(a),range(D)):
        A=[int(q==i) for q in range(a)]; flat=[int(q==j) for q in range(D)]
        B=[flat[q*b:(q+1)*b] for q in range(c+1)]+[flat[(c+1)*b:]]
        ref=direct(a,b,A,B)
        got=[sum(w[r]*U[r][i]*V[r][j] for r in range(len(U))) for w in W]
        assert got==ref;checked+=O
    return {'map':'same-left polynomial packing, not matrix multiplication','a':a,'b':b,'c':c,'t':t,
        'rank_upper_bound':len(U),'verified_coefficients':checked,'U':U,'V':V,'W':W}

if __name__=='__main__':
    result=factors(2,2,5,1)
    print(json.dumps(result,default=str,indent=2))
