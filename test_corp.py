import Tor10 as Tt

import numpy as np 
import torch as tor
import copy


x = tor.ones(3,3)
print(x)

y = Tt.From_torch(x,N_inbond=1,labels=[4,5])
y.Print_diagram()
print(y)

x2 = tor.ones(3,4,requires_grad=True)
print(x2)
y2 = Tt.From_torch(x2,N_inbond=1)
print(y2.requires_grad())

