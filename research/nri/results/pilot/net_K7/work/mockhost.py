import torch, sys
sys.path.insert(0,'.')
from arch import load_net
class Host:
    def __init__(self):
        self.model=load_net(); d=torch.load('inputs.pt'); self.T=d['tokens'][:14000]; self.C=d['ch2'][:14000]
    def sampler(self,n,generator):
        i=torch.randint(0,len(self.T),(n,),generator=generator); return self.T[i],self.C[i]
    def site(self,name,tokens,ch2):
        assert name=='resid_pre_mlp'
        with torch.no_grad(): return self.model.resid_pre_mlp(tokens,ch2)
