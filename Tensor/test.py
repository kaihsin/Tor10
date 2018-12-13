import Ttensor as Tt
import numpy as np 
import torch as tor
import copy


## Example for Bond:
bd_x = Tt.Bond(Tt.BD_IN,3)
bd_y = Tt.Bond(Tt.BD_OUT,4)
bd_z = Tt.Bond(Tt.BD_IN,3)
print(bd_x)
print(bd_y)
print(bd_x==bd_z) ## This should be true
print(bd_x is bd_z) ## This should be false
print(bd_x==bd_y) ## This should be false



"""
#device = tor.device("cuda:0")
x = Tt.UniTensor(D_IN=[5],D_OUT=[5,3],label=[4,2,5],dtype=tor.float64,device=tor.device("cpu"))
y = Tt.UniTensor(D_IN=[2],D_OUT=[3],label=[1,3],dtype=tor.float64,device=tor.device("cpu"))

#print(len(x))
print(x.shape())
print(x)
x.Print_diagram()
print(x.label)

c = Tt.Contract(x,y)
print(c)
c.Print_diagram()

c.Rand()
print(c)
"""





"""
print(x)
print(y)

y[1,2] = 1
x[0,1] = 4

print(y)

y *= 2

print(x)
print(y)

c = x+y
print(c)
"""
