import torch, collections, itertools
d = torch.load('inputs.pt'); T=d['tokens'].tolist(); P=torch.load('work/base.pt')['pred'].tolist()
def det(keyf):
    g=collections.defaultdict(collections.Counter)
    for t,p in zip(T,P): g[keyf(t)][p]+=1
    return sum(c.most_common(1)[0][1] for c in g.values())/len(T), len(g)
def opinfo(t):
    k=[i for i in range(1,5) if t[i]>=11][0]; return k,t[k]
print('t0,t5,op', det(lambda t:(t[0],t[5],opinfo(t)[1])))
print('t0,t5,op,pos', det(lambda t:(t[0],t[5],opinfo(t))))
for a,b in itertools.combinations(range(6),2):
    print(a,b, det(lambda t:(t[a],t[b],opinfo(t)[1])))
print('neighbors', det(lambda t:(t[opinfo(t)[0]-1],t[opinfo(t)[0]+1],opinfo(t)[1])))
