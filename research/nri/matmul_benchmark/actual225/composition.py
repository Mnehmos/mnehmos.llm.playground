"""Pass 3.2/3.4 and same-left packing composition, exact rational reference.
Not the complete Lemma 4.2 round or a qualifying matrix-multiplication kernel.
Vectors are sparse coordinate dictionaries; no global coefficient arrays.
"""
from fractions import Fraction as Q
from functools import lru_cache
from math import comb, prod
from itertools import product
from collections import Counter
from lazy import Primal
from graph import BudgetExceeded


def add(out,k,v):
    if not v: return
    x=out.get(k,Q(0))+v
    if x: out[k]=x
    else: out.pop(k,None)

class Budget:
    def __init__(self,work=3_000_000,live=100_000):
        self.limit=work; self.live=live; self.stats=Counter()
    def tick(self,key,n=1):
        self.stats[key]+=n; self.stats['work']+=n
        if self.stats['work']>self.limit: raise BudgetExceeded('work budget')
    def check(self,v):
        self.stats['peak_vector']=max(self.stats['peak_vector'],len(v))
        if len(v)>self.live: raise BudgetExceeded('live entries')

class Balanced:
    """Rank/unrank balanced target words without enumerating the Cartesian product."""
    def __init__(self,c,d):
        if c<1 or d<1: raise ValueError('positive c,d')
        self.c=c; self.d=d; self.p=2*d; self.m=c+2
    def isfull(self,axis,label):
        return (label<=self.c)==(axis%2==0)
    @lru_cache(maxsize=8192)
    def count(self,pos,need):
        n=self.p-pos
        if not 0<=need<=n:return 0
        p=(n+int(pos%2==0))//2 if n%2 else n//2
        q=n-p; f=self.c+1
        return sum(comb(p,h)*comb(q,need-h)*f**(h+q-(need-h)) for h in range(max(0,need-q),min(p,need)+1))
    @property
    def total(self): return self.count(0,self.d)
    @property
    def promised(self): return (self.m**self.p+self.d)//(self.d+1)
    def unrank(self,r):
        if not 0<=r<self.total:raise IndexError(r)
        need=self.d; w=[]
        for pos in range(self.p):
            for label in range(self.m):
                f=int(self.isfull(pos,label)); k=self.count(pos+1,need-f)
                if r<k:w.append(label);need-=f;break
                r-=k
        return tuple(w)
    def rank(self,w):
        if len(w)!=self.p or any(not 0<=x<self.m for x in w) or sum(self.isfull(i,x) for i,x in enumerate(w))!=self.d:raise ValueError('not balanced')
        need=self.d;r=0
        for pos,label in enumerate(w):
            for l in range(label):r+=self.count(pos+1,need-int(self.isfull(pos,l)))
            need-=int(self.isfull(pos,label))
        return r

@lru_cache(maxsize=128)
def local_maps(g,eps):
    """Sparse E columns and D columns; transposes used for the dual."""
    a,b,c,t,s,l=g.a,g.b,g.c,g.t,g.s,g.l
    E={};D={};ET={};DT={}
    def edge(M,T,i,j,v):
        if v:M.setdefault(i,[]).append((j,Q(v)));T.setdefault(j,[]).append((i,Q(v)))
    for p in range(c+1):
        for i in range(b):
            if p<c:edge(E,ET,(p,i),(p,i),1)
            if p>0:edge(E,ET,(p,i),(p-1,t+i),-1)
    for v in range(s):
        j,i=divmod(v,t);edge(E,ET,(c+1,v),(j,b+i),eps)
    for j,h in product(range(c),range(l)):
        for p in range(c+1):
            v=h-(p-j)*t
            if 0<=v<s:
                z=int(v+p*t<s)-int(j<p)
                edge(D,DT,(j,h),(p,v),z)
        v=j*t+h-s
        if 0<=v<b:edge(D,DT,(j,h),(c+1,v),1/eps)
    return E,D,ET,DT

class Packing:
    def __init__(self,a,b,c,t,d=1):
        self.g=Primal(a,b,c,t);self.words=Balanced(c,d);self.p=2*d
        self.source_count=c**self.p
    def transform(self,vec,eps,decode,budget):
        eps=Q(eps)
        if not eps:raise ValueError('nonzero parameter required')
        E,D,ET,DT=local_maps(self.g,eps)
        for axis in range(self.p):
            table=(DT if axis%2 else E) if not decode else (ET if axis%2 else D)
            nxt={}
            for k,x in vec.items():
                for dest,v in table.get(k[axis],()):
                    budget.tick('linear_edges');add(nxt,k[:axis]+(dest,)+k[axis+1:],x*v)
                    budget.check(nxt)
            vec=nxt
        return vec
    def encode(self,B,eps,budget): return self.transform(B,eps,False,budget)
    def decode(self,C,eps,budget): return self.transform(C,eps,True,budget)
    def source_product(self,A,B,budget):
        """Direct source evaluator, polynomial coefficient pairs; counts scalar products."""
        g=self.g;C={}
        for i,x in A.items():
            for j,y in B.items():
                budget.tick('source_pairs');k=[]
                for axis,(u,(block,v)) in enumerate(zip(i,j)):
                    z=u+v-(g.a-1 if axis%2 else 0)
                    if not 0<=z<(g.b+g.t if axis%2 else g.l):break
                    k.append((block,z))
                else:
                    add(C,tuple(k),x*y);budget.stats['scalar_products']+=1;budget.check(C)
        return C
    def evaluate(self,A,B,eps,budget,source=None):
        enc=self.encode(B,eps,budget)
        C=(source or self.source_product)(A,enc,budget)
        return self.decode(C,eps,budget)
    def target_oracle(self,A,B):
        """Independent direct target formula (no packing maps)."""
        g=self.g;C={}
        for i,x in A.items():
            for j,y in B.items():
                k=[]
                for axis,(u,(block,v)) in enumerate(zip(i,j)):
                    f=self.words.isfull(axis,block)
                    z=u+v-(0 if f else g.a-1)
                    if not 0<=z<(g.s if f else g.b):break
                    k.append((block,z))
                else:add(C,tuple(k),x*y)
        return C
    def target_coords(self,w,output=False):
        g=self.g
        return product(*(range((g.s if self.words.isfull(i,j) else g.b) if output else (g.b if self.words.isfull(i,j) else g.s)) for i,j in enumerate(w)))
    def embed(self,w,vec):return {tuple(zip(w,k)):v for k,v in vec.items() if v}
    def extract(self,w,vec):return {tuple(v for _,v in k):x for k,x in vec.items() if tuple(j for j,_ in k)==w}

def constant(eval_fn,degree,budget):
    total={}
    for e in range(1,degree+2):
        v=eval_fn(Q(e)); budget.stats['interpolation_nodes']+=1
        weight=(-1)**(e-1)*comb(degree+1,e)
        for k,x in v.items():budget.tick('interpolation_edges');add(total,k,weight*x);budget.check(total)
    return total

def lagrange_at(nodes,k,x):
    return prod((x-v)/(nodes[k]-v) for j,v in enumerate(nodes) if j!=k)

def mix(vectors,weights,budget):
    out={}
    for v,w in zip(vectors,weights):
        if w:
            for k,x in v.items():budget.tick('mix_edges');add(out,k,w*x);budget.check(out)
    return out

def independent(P,words,A,B,budget):
    """Lemma 3.4 at scalar base, w independent lefts and w columns per target.
    Global parameter epsilon; inner packing parameter epsilon**w.
    Degree <= P.p*w + 2*w-2 after cancellation. No early exactification.
    B[i][j] is one right polynomial for target i, column j.
    """
    w=len(words)
    if not w or len(A)!=w or len(B)!=w or any(len(v)!=w for v in B) or len(set(words))!=w:raise ValueError('shape mismatch')
    for word in words:P.words.rank(word)
    nodes=tuple(map(Q,range(2*w-1))); left_nodes=nodes[:w]
    left=[mix(A,[lagrange_at(left_nodes,i,k) for i in range(w)],budget) for k in nodes]
    def evaluate(eps):
        values=[]
        for k,Ak in zip(nodes,left):
            enc={}
            for i,word in enumerate(words):
                ns=tuple(Q(i)+eps*j for j in range(w))
                v=mix(B[i],[lagrange_at(ns,j,k) for j in range(w)],budget)
                enc.update(P.embed(word,v))
            values.append(P.evaluate(Ak,enc,eps**w,budget))
            budget.stats['packing_calls']+=1
        out={}
        for i,word in enumerate(words):
            slices=[P.extract(word,v) for v in values]
            for j in range(w):
                v=mix(slices,[lagrange_at(nodes,k,Q(i)+eps*j) for k in range(len(nodes))],budget)
                out.update({(i,j,k):x for k,x in v.items()})
        return out
    return constant(evaluate,P.p*w+2*w-2,budget)

class Compose:
    """Refine outer source products by an inner same-left packing.
    Match source word types (Full,Mid,...); inject into distinct inner targets.
    One global epsilon with inner parameter epsilon**(p+1), so outer poles
    cannot erase the inner error's positive degree. Total degree <= p*(p+2).
    """
    def __init__(self,outer,inner):
        x,y=outer.g,inner.g
        if outer.p!=inner.p or (x.a,x.b+x.t)!=(y.a,y.b):raise ValueError('incompatible source/target shapes')
        if y.c+1<x.c:raise ValueError('not enough aligned inner targets')
        self.outer=outer;self.inner=inner;self.p=outer.p
    def evaluate(self,A,B,budget):
        outer,inner=self.outer,self.inner
        def ev(eps):
            def source(A,enc,budget):
                budget.stats['inner_calls']+=1
                out=inner.evaluate(A,enc,eps**(self.p+1),budget)
                return {k:v for k,v in out.items() if all(j<outer.g.c for j,_ in k)}
            return outer.evaluate(A,B,eps,budget,source)
        return constant(ev,self.p*(self.p+2),budget)

class Chain:
    """Arbitrary compatible same-left refinements; leaves execute source products.
    All stages share one formal parameter. The safe degree bound for k stages
    is (p+1)**k-1. Refuse large exactification before starting work.
    """
    def __init__(self,stages,max_nodes=256):
        self.stages=tuple(stages)
        if not self.stages:raise ValueError('empty chain')
        for x,y in zip(self.stages,self.stages[1:]):Compose(x,y)
        self.p=self.stages[0].p;self.degree=(self.p+1)**len(self.stages)-1
        if self.degree+1>max_nodes:raise BudgetExceeded('interpolation node budget')
    def evaluate(self,A,B,budget):
        def ev(eps):
            def descend(level,A,B):
                P=self.stages[level];eta=eps**((self.p+1)**level)
                budget.stats['stage_calls']+=1
                if level+1==len(self.stages):return P.evaluate(A,B,eta,budget)
                def source(A,enc,budget):
                    out=descend(level+1,A,enc)
                    return {k:v for k,v in out.items() if all(j<P.g.c for j,_ in k)}
                return P.evaluate(A,B,eta,budget,source)
            return descend(0,A,B)
        return constant(ev,self.degree,budget)

class SymbolicChain(Chain):
    """Carry exact Laurent degrees through the same chain, extract only degree zero.
    No interpolation evaluations. Degree terms merge by (coordinate, degree).
    This is an alternative evaluation of the same maps, not a better rank bound.
    """
    def __init__(self,stages,max_degree_terms=10**9):
        super().__init__(stages,max_nodes=max_degree_terms)
    def evaluate(self,A,B,budget):
        B={(k,0):Q(v) for k,v in B.items() if v}
        def transform(P,vec,level,decode):
            E,D,ET,DT=local_maps(P.g,Q(1));scale=(self.p+1)**level
            for axis in range(self.p):
                dual=axis%2
                table=(DT if dual else E) if not decode else (ET if dual else D)
                nxt={}
                for (k,e),x in vec.items():
                    src=k[axis]
                    for dest,v in table.get(src,()):
                        special=P.g.c+1
                        if not decode: shift=(-int(src[0]==special) if dual else int(src[0]==special))
                        else: shift=(int(dest[0]==special) if dual else -int(dest[0]==special))
                        budget.tick('symbolic_edges')
                        add(nxt,(k[:axis]+(dest,)+k[axis+1:],e+scale*shift),x*v);budget.check(nxt)
                vec=nxt
            return vec
        def descend(level,B):
            P=self.stages[level];budget.stats['stage_calls']+=1
            enc=transform(P,B,level,False)
            if level+1<len(self.stages):
                C={k:v for k,v in descend(level+1,enc).items() if all(j<P.g.c for j,_ in k[0])}
            else:
                groups={}
                for (k,e),v in enc.items():groups.setdefault(e,{})[k]=v
                C={}
                for e,vec in groups.items():
                    for k,v in P.source_product(A,vec,budget).items():C[k,e]=v;budget.check(C)
            return transform(P,C,level,True)
        out=descend(0,B)
        if any(e<0 for k,e in out):raise ArithmeticError('uncancelled negative powers')
        budget.stats['final_degree_terms']=len(out)
        budget.stats['max_final_degree']=max((e for k,e in out),default=0)
        return {k:v for (k,e),v in out.items() if e==0}
