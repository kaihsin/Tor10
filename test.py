import Tor10 as Tt

import numpy as np 
import torch as tor
import copy


## Example for Bond:
print("Testing for Bond")
bd_x = Tt.Bond(Tt.BD_IN,3)
bd_y = Tt.Bond(Tt.BD_OUT,4)
bd_z = Tt.Bond(Tt.BD_IN,3)
print(bd_x)
print(bd_y)
print(bd_x==bd_z) ## This should be true
print(bd_x is bd_z) ## This should be false
print(bd_x==bd_y) ## This should be false


## Example for Symmetry Bond:
print("Testing symmetry Bond")
bd_sym_x = Tt.Bond(Tt.BD_IN,3,qnums=[[0,-3],[1,-4],[2,5]])
bd_sym_y = Tt.Bond(Tt.BD_OUT,4,qnums=[[-1,0],[2,1],[0,1],[2,1]])
print(bd_sym_x)
print(bd_sym_y)


## Testing combine sym Bond:
bd_sym_x = Tt.Bond(Tt.BD_IN,3,qnums=[[0,-3],[1,-4],[2,5]])
bd_sym_y = Tt.Bond(Tt.BD_OUT,4,qnums=[[-1,0],[2,1],[0,1],[2,1]])
print("Testing combine sym Bond")
bd_sym_x.combine(bd_sym_y)
print(bd_sym_x)


## Testing combine
bds_x = [Tt.Bond(Tt.BD_IN,5),Tt.Bond(Tt.BD_OUT,5),Tt.Bond(Tt.BD_OUT,3)]
bds_y = [Tt.Bond(Tt.BD_IN,2),Tt.Bond(Tt.BD_OUT,3)]
x = Tt.UniTensor(bonds=bds_x, labels=[4,3,5],dtype=tor.float64,device=tor.device("cpu"))
y = Tt.UniTensor(bonds=bds_y, labels=[1,5]  ,dtype=tor.float64,device=tor.device("cpu"))
print(x.shape())
print(x)
x.Print_diagram()
print(x.labels)
y.Print_diagram()
print(y.labels)


## Testing Contract:
bds_x = [Tt.Bond(Tt.BD_IN,5),Tt.Bond(Tt.BD_OUT,5),Tt.Bond(Tt.BD_OUT,3)]
bds_y = [Tt.Bond(Tt.BD_IN,2),Tt.Bond(Tt.BD_OUT,3)]
x = Tt.UniTensor(bonds=bds_x, labels=[4,3,5],dtype=tor.float64,device=tor.device("cpu"))
y = Tt.UniTensor(bonds=bds_y, labels=[1,5]  ,dtype=tor.float64,device=tor.device("cpu"))
c = Tt.Contract(x,y)
print(c)
c.Print_diagram()

bds_x2 = [Tt.Bond(Tt.BD_IN,5),Tt.Bond(Tt.BD_OUT,2),Tt.Bond(Tt.BD_OUT,3)]
bds_y2 = [Tt.Bond(Tt.BD_OUT,2),Tt.Bond(Tt.BD_OUT,3)]
x = Tt.UniTensor(bonds=bds_x2, labels=[4,1,5],dtype=tor.float64,device=tor.device("cpu"))
y = Tt.UniTensor(bonds=bds_y2, labels=[1,5]  ,dtype=tor.float64,device=tor.device("cpu"))
c = Tt.Contract(x,y)
print(c)
c.Print_diagram()

bds_x3 = [Tt.Bond(Tt.BD_OUT,2),Tt.Bond(Tt.BD_OUT,3)]
bds_y3 = [Tt.Bond(Tt.BD_OUT,2),Tt.Bond(Tt.BD_OUT,3)]
x = Tt.UniTensor(bonds=bds_x3, labels=[1,5],dtype=tor.float64,device=tor.device("cpu"))
y = Tt.UniTensor(bonds=bds_y3, labels=[1,5]  ,dtype=tor.float64,device=tor.device("cpu"))
c = Tt.Contract(x,y)
print(c)
#c.Print_diagram()

exit(1)


y[1,2] = 1
x[0,1] = 4
y *= 2

print("------------")
## example for reshape
## Note that reshape on a non-contiguous tensor will have warning. This is the same as pytorch.
bds_x = [Tt.Bond(Tt.BD_IN,6),Tt.Bond(Tt.BD_OUT,5),Tt.Bond(Tt.BD_OUT,3)]
x = Tt.UniTensor(bonds=bds_x, labels=[4,3,5],dtype=tor.float64,device=tor.device("cpu"))
x.Print_diagram()
x.Reshape([2,3,5,3],new_labels=[1,2,3,-1],N_inbond=2)
x.Print_diagram()


print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
bds_x = [Tt.Bond(Tt.BD_IN,6),Tt.Bond(Tt.BD_OUT,5),Tt.Bond(Tt.BD_OUT,3)]
x = Tt.UniTensor(bonds=bds_x, labels=[4,3,5])
y = Tt.UniTensor(bonds=bds_x, labels=[4,3,5])
x.Print_diagram()
x.Permute([0,2,1],2)
x.Print_diagram()

y.Permute([3,4,5],2,by_label=True)
y.Print_diagram()


## example of permute:
print("===========")
c.Print_diagram()
print(c.shape())
print(c.labels)
c.Permute([0,2,1],1)
print(c.is_contiguous()) ## This should be false. The virtual permute is taking action.
c.Print_diagram()

## example of Chain_matmul
print("pppppppppppppppppp")
a = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,3),Tt.Bond(Tt.BD_OUT,4)],labels=[0,1])
b = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,4),Tt.Bond(Tt.BD_OUT,5)],labels=[2,3])
c = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,5),Tt.Bond(Tt.BD_OUT,6)],labels=[4,6])   
d = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,6),Tt.Bond(Tt.BD_OUT,2)],labels=[5,-1])
f = Tt.Chain_matmul(a,b,c,d)
f.Print_diagram()


print("Testing determinant==========")
a = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,3),Tt.Bond(Tt.BD_OUT,3)],labels=[0,1])
a.SetElem([4,-3,0,\
           2,-1,2,\
           1, 5,7])
print(a)
out = Tt.Det(a)
print(out)
a = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,3),Tt.Bond(Tt.BD_OUT,3)],labels=[0,1],is_diag=True)
a.SetElem([1,2,3])
print(a)
out = Tt.Det(a)
print(out)

exit(1)
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




