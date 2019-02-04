import Tor10 as Tt

import numpy as np 
import torch as tor
import copy



bds_x = [Tt.Bond(Tt.BD_IN,5),Tt.Bond(Tt.BD_OUT,5),Tt.Bond(Tt.BD_OUT,3)]
x = Tt.UniTensor(bonds=bds_x, labels=[4,3,5])
print(x.requires_grad())
x.requires_grad(True)
print(x.requires_grad())
x.requires_grad(False)
print(x.requires_grad())
print("===================")


bds_x = [Tt.Bond(Tt.BD_IN,5),Tt.Bond(Tt.BD_OUT,5),Tt.Bond(Tt.BD_OUT,3)]
bds_y = [Tt.Bond(Tt.BD_IN,2),Tt.Bond(Tt.BD_OUT,3)]
x = Tt.UniTensor(bonds=bds_x, labels=[4,3,5],dtype=tor.float64,device=tor.device("cpu"),requires_grad=True)
y = Tt.UniTensor(bonds=bds_y, labels=[1,5]  ,dtype=tor.float64,device=tor.device("cpu"),requires_grad=True)
print(x.requires_grad())
print(y.requires_grad())

c = Tt.Contract(x,y)
#print(c)
print(c.requires_grad())
c.Print_diagram()


