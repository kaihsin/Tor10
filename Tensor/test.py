import Ttensor as Tt
import numpy as np 
import torch as tor
import copy

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



print(c)


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
