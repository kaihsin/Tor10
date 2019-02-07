import Tor10 

import numpy as np 
import torch as tor
import copy


"""
x = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,2),Tor10.Bond(Tor10.BD_OUT,2)],requires_grad=True) 
print(x)
y = (x+4)**2
print(y)
out = Tor10.Mean(y)
print(out)
out.backward()
print(x.grad())
"""


import torch.nn as nn
import torch.nn.functional as F

m = Tor10.nn.Linear(20, 30)
iput = Tor10.From_torch(tor.randn(128, 20),N_inbond=1)
oput = m(iput)
print(oput.shape())
exit(1)
        
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        
        ## Customize and register the parameter.
        self.P1 = Tor10.nn.Parameter(Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,2),Tor10.Bond(Tor10.BD_OUT,2)]))
        self.P2 = Tor10.nn.Parameter(Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,2),Tor10.Bond(Tor10.BD_OUT,2)]))

        self.P3 = Tor10.nn.Linear(2,2)
        
    def forward(self,x):
        x2 = self.P3(x)
        y = Tor10.Matmul(Tor10.Matmul(x,self.P1),self.P2)
        return y


x = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,2),Tor10.Bond(Tor10.BD_OUT,2)])
md = Model()
print(list(md.parameters()))
exit(1)


md2= Model2()
out2 = md2.forward(x.Storage)
print(list(md2.parameters()))


