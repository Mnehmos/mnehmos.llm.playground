import torch, sys
sys.path.insert(0,'.')
from arch import load_net
class Host:
    def __init__(self):
        self.model=load_net(); d=torch.load('inputs.pt'); self.T=d['tokens']; self.C=d['ch2']
    def sampler(self,n,generator):
        i=torch.randint(0,len(self.T),(n,),generator=generator); return self.T[i].clone(), self.C[i].clone()
    def site(self,name,tokens,ch2):
        assert name=='resid_pre_mlp'; return self.model.resid_pre_mlp(tokens,ch2)
