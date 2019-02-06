import Tor10 

import numpy as np 
import torch as tor
import copy



x = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,2),Tor10.Bond(Tor10.BD_OUT,2)],requires_grad=True) 
print(x)
y = (x+4)**2
print(y)
out = Tor10.Mean(y)
print(out)
out.backward()
print(x.grad())

