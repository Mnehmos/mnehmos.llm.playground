import json, random, time, sys, platform
from itertools import product
from composition import *
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
rng=random.Random(307)
def vec(keys): return {k:Q(rng.choice([-3,-2,-1,1,2,3]),rng.choice([1,2,3])) for k in keys}
def timed(fn):
    t=time.perf_counter();z=fn();return z,(time.perf_counter()-t)*1000

def verify():
    ranks=0; rank_rows=[]
    for c,d in [(2,1),(5,1),(2,2),(5,2),(2,3)]:
        b=Balanced(c,d); ref=[w for w in product(range(c+2),repeat=2*d) if sum(b.isfull(i,j) for i,j in enumerate(w))==d]
        assert b.total==len(ref)>=b.promised
        for k,w in enumerate(ref):assert b.unrank(k)==w and b.rank(w)==k;ranks+=1
    for d in [8,32,128]:
        b=Balanced(5,d); rs=[0,b.total//2,b.total-1]; _,ms=timed(lambda:[(b.rank(b.unrank(r))==r) or (_ for _ in ()).throw(AssertionError()) for r in rs]);rank_rows.append(dict(d=d,digits=len(str(b.total)),three_rank_roundtrips_ms=ms));ranks+=3
    rows=[];coeff=0
    for a,b,c,t in [(1,1,2,1),(2,2,5,1),(2,3,4,2),(3,2,2,3)]:
        p=Packing(a,b,c,t);words=[p.words.unrank(i) for i in range(p.words.total)]
        A=vec(product(range(a),repeat=2));B={}
        for w in words:B.update(p.embed(w,vec(p.target_coords(w))))
        ref=p.target_oracle(A,B);budget=Budget();z,ms=timed(lambda:constant(lambda e:p.evaluate(A,B,e,budget),p.p,budget))
        assert z==ref;n=sum(len(list(p.target_coords(w,True))) for w in words);coeff+=n
        rows.append(dict(base=[a,b,c,t],balanced=len(words),checked=n,ms=ms,stats=dict(budget.stats)))
    indep=[];ic=0
    for a,b,c,t,w in [(1,1,2,1,8),(2,2,5,1,2),(2,2,5,1,3),(2,3,4,2,2)]:
        p=Packing(a,b,c,t);ids=list(range(w-1))+[p.words.total-1];words=[p.words.unrank(i) for i in ids]
        A=[vec(product(range(a),repeat=2)) for _ in words]; B=[[vec(p.target_coords(word)) for j in range(w)] for word in words]
        ref={};n=0
        for i,word in enumerate(words):
            for j in range(w):
                z=p.extract(word,p.target_oracle(A[i],p.embed(word,B[i][j])))
                ref.update({(i,j,k):v for k,v in z.items()});n+=len(list(p.target_coords(word,True)))
        bud=Budget(work=10_000_000);z,ms=timed(lambda:independent(p,words,A,B,bud));assert z==ref;ic+=n
        indep.append(dict(base=[a,b,c,t],w=w,checked=n,ms=ms,stats=dict(bud.stats)))
    outer=Packing(2,2,5,1); inner=Packing(2,3,7,1); chain=Compose(outer,inner);comp=[];cc=0
    for mode in ['sparse','dense','all_balanced','zero']:
        words=[outer.words.unrank(i) for i in ([0,outer.words.total-1] if mode!='all_balanced' else range(outer.words.total))]
        A=vec(product(range(2),repeat=2));B={}
        for word in words:
            coords=list(outer.target_coords(word)); B.update(outer.embed(word,vec(coords[:2] if mode=='sparse' else coords)))
        if mode=='zero':A={}
        ref=outer.target_oracle(A,B);bud=Budget();z,ms=timed(lambda:chain.evaluate(A,B,bud));assert z==ref
        base=Budget();direct_z,base_ms=timed(lambda:constant(lambda e:outer.evaluate(A,B,e,base),outer.p,base));assert direct_z==ref
        n=sum(len(list(outer.target_coords(word,True))) for word in words);cc+=2*n
        sb=Budget();sz,sms=timed(lambda:SymbolicChain([outer,inner]).evaluate(A,B,sb));assert sz==ref
        comp.append(dict(mode=mode,checked=n,composed_ms=ms,symbolic_ms=sms,symbolic_stats=dict(sb.stats),single_ms=base_ms,composed_stats=dict(bud.stats),single_stats=dict(base.stats)))
    guards=0
    for f in [lambda:Compose(outer,Packing(2,2,5,1)),lambda:chain.evaluate({(0,0):1},outer.embed(outer.words.unrank(0),{(0,0):1}),Budget(work=1))]:
        try:f()
        except (ValueError,BudgetExceeded):guards+=1
        else:raise AssertionError('guard')
    chain3=Chain([outer,inner,Packing(2,4,9,1)]); words=[outer.words.unrank(0),outer.words.unrank(outer.words.total-1)]
    A=vec(product(range(2),repeat=2)); B={}
    for word in words:B.update(outer.embed(word,vec(outer.target_coords(word))))
    bud=Budget();z,ms=timed(lambda:chain3.evaluate(A,B,bud));assert z==outer.target_oracle(A,B)
    sb=Budget();sz,sms=timed(lambda:SymbolicChain([outer,inner,Packing(2,4,9,1)]).evaluate(A,B,sb));assert sz==z
    chain3_result=dict(depth=3,degree=chain3.degree,checked=12,ms=ms,stats=dict(bud.stats),symbolic_ms=sms,symbolic_stats=dict(sb.stats));cc+=24
    try:Chain([outer,inner,Packing(2,4,9,1)],max_nodes=10)
    except BudgetExceeded:guards+=1
    else:raise AssertionError('node guard')
    return dict(chain3=chain3_result,python=platform.python_version(),balanced_rank_checks=ranks,large_rank=rank_rows,amplification_output_checks=coeff,amplification=rows,independent_output_checks=ic,independent=indep,composition_output_checks=cc,composition=comp,guards=guards)
if __name__=='__main__': print(json.dumps(verify(),indent=2))
