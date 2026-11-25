from composition import *
import time,json,random

def run(A,B,outer):
    ref=outer.target_oracle(A,B); rows=[]
    for depth in [1,2,3,4,6,8,12]:
        bud=Budget();t=time.perf_counter()
        stages=[Packing(2,b,2*b+1,1) for b in range(2,2+depth)]
        try:
            z=SymbolicChain(stages).evaluate(A,B,bud);assert z==ref
            rows.append(dict(depth=depth,ms=(time.perf_counter()-t)*1000,safe_interpolation_nodes=3**depth,stats=dict(bud.stats)))
        except BudgetExceeded as e:
            rows.append(dict(depth=depth,status=str(e),stats=dict(bud.stats)));break
    return rows

if __name__=='__main__':
    outer=Packing(2,2,5,1);w=outer.words.unrank(0)
    sparse=run({(0,0):Q(2,3),(1,1):Q(-3,5)},outer.embed(w,{(0,0):Q(1),(1,2):Q(2)}),outer)
    rng=random.Random(311)
    A={k:Q(rng.choice([-3,-2,-1,1,2,3]),3) for k in product(range(2),repeat=2)}
    B={}
    for i in range(outer.words.total):
        word=outer.words.unrank(i)
        B.update(outer.embed(word,{k:Q(rng.choice([-3,-2,-1,1,2,3]),2) for k in outer.target_coords(word)}))
    dense=run(A,B,outer)
    print(json.dumps(dict(sparse=sparse,dense=dense),indent=2))
