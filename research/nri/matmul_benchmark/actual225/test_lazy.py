from lazy import Primal,Dual,ExactPower,Q,lagrange
from extract_factors import factors
from constructive import direct
from itertools import product
import random, time, tracemalloc, json, platform
import sys
if hasattr(sys,"set_int_max_str_digits"): sys.set_int_max_str_digits(0)


def direct_entry(g,i,j,k):
    if isinstance(g,Dual): return direct_entry(g.base,g.dims[0]-1-i,k,j)
    a,b,c,t=g.a,g.b,g.c,g.t
    A=[int(q==i) for q in range(a)]; flat=[int(q==j) for q in range(g.dims[1])]
    B=[flat[q*b:(q+1)*b] for q in range(c+1)]+[flat[(c+1)*b:]]
    return direct(a,b,A,B)[k]


def expected(P,i,j,k):
    ix=[i,j,k]; v=1
    for p in range(P.power-1,-1,-1):
        g=P.dual if P.alternate and p%2 else P.base
        digits=[]
        for l in range(3): ix[l],d=divmod(ix[l],g.dims[l]); digits.append(d)
        v*=direct_entry(g,*digits)
    return v


def check_factors(args):
    P=ExactPower(Primal(*args),1); ref=factors(*args); checks=0
    for l,key in enumerate(('U','V','W')):
        for r,i in product(range(P.terms),range(P.dims[l])):
            assert P.coefficient(l,r,i)==(ref[key][i][r] if l==2 else ref[key][r][i]); checks+=1
    return checks


def run():
    g=Primal(2,2,5,1)
    checks=check_factors((2,2,5,1))
    count={}; rng=random.Random(307)
    for power,alternate in [(1,False),(2,False),(2,True),(3,True)]:
        p=ExactPower(Primal(1,1,2,1),power,alternate)
        if power<=2: cases=list(product(*(range(d) for d in p.dims)))
        else: cases=[tuple(rng.randrange(d) for d in p.dims) for _ in range(100)]
        for i,j,k in cases: assert p.entry(i,j,k)==expected(p,i,j,k)
        count[f'tiny_{power}_{alternate}']=len(cases)
    for alt in (False,True):
        p=ExactPower(g,2,alt); cases=[tuple(rng.randrange(d) for d in p.dims) for _ in range(24)]
        for _ in range(12):
            coords=[0,0,0]
            for k in range(2):
                gg=p.dual if alt and k%2 else g
                while True:
                    d=[rng.randrange(n) for n in gg.dims]
                    if direct_entry(gg,*d):break
                coords=[x*n+y for x,n,y in zip(coords,gg.dims,d)]
            cases.append(tuple(coords))
        for i,j,k in cases: assert p.entry(i,j,k)==expected(p,i,j,k)
        count[f'actual_2_{alt}']=len(cases)
    rows=[]
    for power in (1,2,4,8,16,32,128,4096):
        lagrange.cache_clear();tracemalloc.start(); start=time.perf_counter()
        p=ExactPower(g,power,True)
        queries=[(0,0,0),(1,0,0),(2,0,0),(0,p.terms-1,0),(1,p.terms-1,p.dims[1]-1),(2,p.terms-1,p.dims[2]-1)]
        vals=[p.coefficient(*q) for q in queries]
        elapsed=time.perf_counter()-start; _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        rows.append({'power':power,'rank_digits':len(str(p.terms)),'dense_slots_digits':len(str(p.dense_slots)),'dense_slots':str(p.dense_slots) if power<=32 else None,'six_queries_ms':elapsed*1000,'peak_traced_bytes':peak,'nonzero_queries':sum(bool(v) for v in vals)})
    extra={str(args):check_factors(args) for args in [(1,2,3,2),(2,3,4,2),(3,2,2,3),(4,1,1,5)]}
    return {'python':platform.python_version(),'dense_factor_checks':checks,'extra_factor_checks':extra,'contraction_checks':count,'total_contraction_checks':sum(count.values()),'scaling':rows,'note':'Dense slots count scalar coefficient entries, not actual allocated bytes; memory excludes interpreter/imports; timings include descriptor construction and cold cache.'}
if __name__=='__main__':print(json.dumps(run(),indent=2))
