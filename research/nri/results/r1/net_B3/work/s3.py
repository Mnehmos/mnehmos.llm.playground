import torch, collections, itertools
d=torch.load('inputs.pt'); T=d['tokens'].tolist(); P=torch.load('work/base.pt')['pred'].tolist()
def purity(keys,ys):
    g=collections.defaultdict(collections.Counter)
    for k,y in zip(keys,ys): g[k][y]+=1
    return sum(c.most_common(1)[0][1] for c in g.values())/len(ys), len(g)
for opos in range(1,5):
  for op in (11,12,13):
    rows=[(t,p) for t,p in zip(T,P) if t[opos]==op]
    nums=[i for i in range(6) if i!=opos]
    res=[]
    for a,b in itertools.combinations(nums,2):
        pu,ng=purity([(t[a],t[b]) for t,_ in rows],[p for _,p in rows])
        res.append((round(pu,3),a,b))
    for a in nums:
        pu,ng=purity([t[a] for t,_ in rows],[p for _,p in rows]); res.append((round(pu,3),a))
    res.sort(reverse=True)
    print(opos,op,len(rows),res[:4])
