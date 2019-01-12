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

bd_sym_x = Tt.Bond(Tt.BD_IN,3,qnums=[0,1,2])
bd_sym_y = Tt.Bond(Tt.BD_OUT,4,qnums=[-1,2,0,2])
print(bd_sym_x)
print(bd_sym_y)


device = tor.device("cuda:0")
bds_x = [Tt.Bond(Tt.BD_IN,5),Tt.Bond(Tt.BD_OUT,5),Tt.Bond(Tt.BD_OUT,3)]
bds_y = [Tt.Bond(Tt.BD_IN,2),Tt.Bond(Tt.BD_OUT,3)]
x = Tt.UniTensor(bonds=bds_x, labels=[4,3,5],dtype=tor.float64,device=tor.device("cpu"))
y = Tt.UniTensor(bonds=bds_y, labels=[1,5]  ,dtype=tor.float64,device=tor.device("cpu"))
exit(1)
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


## example for reshape
## Note that reshape on a non-contiguous tensor will have warning. This is the same as pytorch.
x.Print_diagram()
x.Reshape([5,3,5],new_labels=[1,2,3],N_inbond=2)
x.Print_diagram()



## example of permute:
print("===========")
c.Print_diagram()
print(c.shape())
print(c.labels)
c.Permute([0,2,1],1)
print(c.is_contiguous()) ## This should be false. The virtual permute is taking action.
c.Print_diagram()





""
c.CombineBonds([1,3]) ## The CombineBonds implicitly have contiguous only at the last stage when moving memory is needed. 
c.Print_diagram()
print(c.shape())
print(c.labels)
""




""
## Test Svd:
## ------------------------
u,s,v = c.Svd()
u.Print_diagram()
s.Print_diagram()
v.Print_diagram()

## Test chain_matmul , derive from pytorch
## -----------------------
out = Tt.Chain_matmul(u,s,v)
print(out - c) # this should be all zeros.

## Test contiguous
## -----------------------
out.Contiguous()
print(out.is_contiguous()) #this should be true

"""
## Test I/O
## ----------------------
Tt.Save(out,"test.uni10")
out2 = Tt.Load("test.uni10")

print(out2)

print(out==out2)
"""




