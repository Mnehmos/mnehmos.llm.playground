from graph import Graph,BudgetExceeded,Q
from lazy import Primal,Dual,ExactPower
from constructive import direct
import random,time,json,tracemalloc,platform
from itertools import product

def oracle_local(g,i,j,k):
    if isinstance(g,Dual): return oracle_local(g.base,g.dims[0]-1-i,k,j)
    a,b,c=g.a,g.b,g.c
    A=[int(q==i) for q in range(a)]; flat=[int(q==j) for q in range(g.dims[1])]
    B=[flat[q*b:(q+1)*b] for q in range(c+1)]+[flat[(c+1)*b:]]
    return direct(a,b,A,B)[k]

def oracle_entry(graph,i,j,k):
    digs=[graph.digits(x,l) for l,x in enumerate((i,j,k))]; v=1
    for g,a,b,c in zip(graph.gs,*digs):
        v*=oracle_local(g,a,b,c)
        if not v:break
    return v

def oracle(graph,A,B,outputs):
    return {k:sum(a*b*oracle_entry(graph,i,j,k) for i,a in A.items() for j,b in B.items()) for k in outputs}

def sparse(rng,n,count):return {i:Q(rng.choice([-3,-2,-1,1,2,3]),rng.choice([1,2,3])) for i in rng.sample(range(n),min(n,count))}
def timed(fn):
    t=time.perf_counter(); v=fn(); return v,(time.perf_counter()-t)*1000

def run():
    rng=random.Random(307); checks=0; rows=[]
    for g in [Primal(1,1,2,1),Primal(2,2,5,1),Primal(2,3,4,2)]:
        for d,alt in [(1,False),(2,False),(2,True)]:
            p=ExactPower(g,d,alt); graph=Graph(p)
            A=sparse(rng,p.dims[0],5); B=sparse(rng,p.dims[1],8)
            outs=list(range(p.dims[2])); ref=oracle(graph,A,B,outs)
            got,ms=timed(lambda:graph.full(A,B)); assert all(got.get(k,0)==v for k,v in ref.items());checks+=len(outs)
            sg=Graph(p); selected=rng.sample(outs,min(12,len(outs)))
            v,sms=timed(lambda:sg.sparse(A,B,selected)); assert all(v[k]==ref[k] for k in selected);checks+=len(selected)
            eg=Graph(p); i,j,k=0,0,0
            x,baseline_ms=timed(lambda:p.entry(i,j,k)); y,entry_ms=timed(lambda:eg.entry(i,j,k)); assert x==y==oracle_entry(eg,i,j,k);checks+=1
            rows.append(dict(base=[g.a,g.b,g.c,g.t],power=d,alternate=alt,dims=p.dims,global_terms=p.terms,full_ms=ms,full_stats=graph.snapshot(),sparse_ms=sms,sparse_stats=sg.snapshot(),entry_baseline_ms=baseline_ms,entry_graph_ms=entry_ms))
    p=ExactPower(Primal(2,2,5,1),1)
    dense_checks=0
    for A,B in [({i:Q((-1)**i,i+1) for i in range(p.dims[0])}, {j:Q((-1)**j,j+1) for j in range(p.dims[1])}), ({0:1,1:-1},{j:1 for j in range(p.dims[1])}), ({},{0:1})]:
        gg=Graph(p); outs=list(range(p.dims[2])); ref=oracle(gg,A,B,outs); got=gg.full(A,B)
        assert all(got.get(k,0)==v for k,v in ref.items())
        assert Graph(p).sparse(A,B,outs)==ref
        dense_checks+=2*len(outs)
    checks+=dense_checks
    big=[]
    for d in (8,32,128,4096):
        p=ExactPower(Primal(2,2,5,1),d,True); graph=Graph(p)
        coords=[0,0,0]
        for g in graph.gs:
            local=next(t for t in product(*(range(n) for n in g.dims)) if oracle_local(g,*t))
            coords=[x*n+y for x,n,y in zip(coords,g.dims,local)]
        assert oracle_entry(graph,*coords)==1
        tracemalloc.start(); v,ms=timed(lambda:graph.sparse({coords[0]:Q(2,3)},{coords[1]:Q(-3,5)},[coords[2]])); _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        assert v[coords[2]]==Q(-2,5);checks+=1
        g=graph.gs[-1]; a,b,c=[x%n for x,n in zip(coords,g.dims)]
        z=next(k for k in range(g.dims[2]) if not oracle_local(g,a,b,k)); out=coords[2]-c+z
        assert graph.sparse({coords[0]:1},{coords[1]:1},[out])[out]==0;checks+=1
        big.append(dict(power=d,global_terms_digits=len(str(p.terms)),ms=ms,peak_traced_bytes=peak,stats=graph.snapshot()))
    try:Graph(ExactPower(Primal(2,2,5,1),2),max_work=1).full({0:1},{0:1})
    except BudgetExceeded:guard=True
    else:raise AssertionError('budget failed')
    return dict(python=platform.python_version(),exact_output_checks=checks,dense_and_zero_checks=dense_checks,budget_guard=guard,small=rows,large=big)
if __name__=='__main__':
    import sys
    if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
    print(json.dumps(run(),indent=2))
