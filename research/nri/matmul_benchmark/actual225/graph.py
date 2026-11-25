"""Exact factorized execution of elementary Pass tensor powers, not a 9/4 matmul kernel."""
from collections import defaultdict
from functools import lru_cache
from itertools import product
from fractions import Fraction as Q
from math import comb, prod
from lazy import Primal, Dual, ExactPower, checked

class BudgetExceeded(RuntimeError): pass

class Graph:
    def __init__(self, power, max_work=2_000_000, max_live=200_000):
        self.p=power; self.max_work=max_work; self.max_live=max_live
        self.gs=tuple(power.dual if power.alternate and p%2 else power.base for p in range(power.power))
        self.dims=power.dims
        self.stats=defaultdict(int)
        self._entry=lru_cache(maxsize=4096)(self._local_entry)
        self._map=lru_cache(maxsize=128)(self._local_map)

    def tick(self,key,n=1):
        self.stats[key]+=n; self.stats['work']+=n
        if self.stats['work']>self.max_work: raise BudgetExceeded('work budget')

    def snapshot(self):
        return dict(self.stats,entry_cache_hits=self._entry.cache_info().hits,entry_cache_misses=self._entry.cache_info().misses)

    def digits(self,i,leg):
        checked(i,self.dims[leg]); out=[]
        for g in reversed(self.gs): i,d=divmod(i,g.dims[leg]); out.append(d)
        return tuple(reversed(out))

    def _local_entry(self,g,i,j,k):
        # Constant of affine local tensor from evaluations at 1 and 2.
        vals=[]
        for eps in (1,2):
            s=Q(0)
            for r in range(g.terms):
                self.tick('local_term_visits')
                u=g.coefficient(0,r,i,eps)
                if not u: continue
                v=g.coefficient(1,r,j,eps)
                if not v: continue
                w=g.coefficient(2,r,k,eps)
                if w: s+=u*v*w; self.stats['local_nonzero_terms']+=1
            vals.append(s)
        return 2*vals[0]-vals[1]

    def entry(self,i,j,k):
        ids=[self.digits(x,l) for l,x in enumerate((i,j,k))]
        return self._entry_digits(*ids)

    def _entry_digits(self,i,j,k):
        v=Q(1)
        for g,a,b,c in zip(self.gs,i,j,k):
            self.tick('factor_visits')
            v*=self._entry(g,a,b,c)
            if not v:
                self.stats['zero_prunes']+=1; break
        return v

    def sparse(self,A,B,outputs):
        """Selected outputs, sparse inputs; never enumerates global rank terms."""
        A=self._sparse_digits(A,0); B=self._sparse_digits(B,1)
        result={}
        for k in dict.fromkeys(outputs):
            self.tick('output_visits')
            kd=self.digits(k,2); total=Q(0)
            for i,a in A.items():
                for j,b in B.items():
                    self.tick('input_pairs')
                    z=self._entry_digits(i,j,kd)
                    if z: total+=a*b*z; self.stats['nonzero_input_pairs']+=1
            result[k]=total
        return result

    def _sparse_digits(self,A,leg):
        if len(A)>self.max_live: raise BudgetExceeded('input budget')
        return {self.digits(i,leg):Q(v) for i,v in A.items() if v}

    def _local_map(self,g,leg,eps):
        # sparse column adjacency for encoder (input -> rank) or decoder (rank -> output).
        d=g.dims[leg]; r=g.terms
        m,n=(r,d) if leg==2 else (d,r)
        cols=[]
        for i in range(m):
            col=[]
            for j in range(n):
                self.tick('local_map_probes')
                v=g.coefficient(leg,i,j,eps) if leg==2 else g.coefficient(leg,j,i,eps)
                if v: col.append((j,v))
            cols.append(tuple(col))
        return tuple(cols)

    def _transform(self,vec,leg,eps):
        # Apply each local factor along one axis. Equal intermediates share a dictionary key.
        for axis,g in enumerate(self.gs):
            cols=self._map(g,leg,eps); nxt={}
            for coord,val in vec.items():
                for j,w in cols[coord[axis]]:
                    self.tick('linear_contributions')
                    idx=coord[:axis]+(j,)+coord[axis+1:]
                    if idx in nxt: self.stats['merged_contributions']+=1
                    z=nxt.get(idx,Q(0))+val*w
                    if z: nxt[idx]=z
                    else: nxt.pop(idx,None); self.stats['cancelled_contributions']+=1
                    if len(nxt)>self.max_live: raise BudgetExceeded('intermediate live-entry budget')
            vec=nxt; self.stats['peak_live_vector']=max(self.stats['peak_live_vector'],len(vec))
        return vec

    def full(self,A,B):
        """Sparse factorized encode/product/decode; exact interpolation across d+1 nodes."""
        A=self._sparse_digits(A,0); B=self._sparse_digits(B,1)
        left=self._transform(A,0,1) # U is independent of epsilon, including dual.
        total={}
        for eps in range(1,self.p.power+2):
            right=self._transform(B,1,eps)
            common=left.keys() & right.keys()
            self.stats['bilinear_products']+=len(common)
            self.tick('rank_intersection_products',len(common))
            joint={i:left[i]*right[i] for i in common}
            decoded=self._transform(joint,2,eps)
            w=(-1)**(eps-1)*comb(self.p.power+1,eps)
            for i,v in decoded.items():
                self.tick('interpolation_contributions')
                z=total.get(i,Q(0))+w*v
                if z: total[i]=z
                else: total.pop(i,None)
                if len(total)>self.max_live: raise BudgetExceeded('output live-entry budget')
        def flatten(coord):
            v=0
            for g,x in zip(self.gs,coord): v=v*g.dims[2]+x
            return v
        return {flatten(k):v for k,v in total.items()}
