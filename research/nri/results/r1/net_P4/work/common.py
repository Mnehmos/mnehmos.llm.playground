import torch
from arch import load_net
net=load_net(); d=torch.load('inputs.pt')
T,C=d['tokens'],d['ch2']
L=T.tolist()
def oppos(t): return [i for i in range(1,5) if t[i]>=11][0]
def A(t): return t[oppos(t)-1]
def B(t): return t[oppos(t)+1]
def OP(t): return t[oppos(t)]
def f(a,b,o): return {11:(a+b)%11,12:(a-b)%11,13:(a*b)%11}[o]
def res(t): return f(A(t),B(t),OP(t))
with torch.no_grad():
    H=net.resid_pre_mlp(T,C); Y=net.head_from_resid(H).argmax(-1)
