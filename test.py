import Tor10 as Tt

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




#device = tor.device("cuda:0")
bds_x = [Tt.Bond(Tt.BD_IN,5),Tt.Bond(Tt.BD_OUT,5),Tt.Bond(Tt.BD_OUT,3)]
bds_y = [Tt.Bond(Tt.BD_IN,2),Tt.Bond(Tt.BD_OUT,3)]
x = Tt.UniTensor(bonds=bds_x, labels=[4,2,5],dtype=tor.float64,device=tor.device("cpu"))
y = Tt.UniTensor(bonds=bds_y, labels=[1,3]  ,dtype=tor.float64,device=tor.device("cpu"))

#print(len(x))
print(x.shape())
print(x)
x.Print_diagram()
print(x.labels)
y.Print_diagram()
print(y.labels)


c = Tt.Contract(x,y)
print(c)
c.Print_diagram()
c.Rand()
print(c)

y[1,2] = 1
x[0,1] = 4
y *= 2
print(y)

print("===========")
c.Print_diagram()
print(c.shape())
print(c.labels)
c.CombineBonds([2,3])
c.Print_diagram()
print(c.shape())
print(c.labels)





