import Tor10 as Tt
import numpy as np 
import torch as tor
import copy


device = tor.device("cuda:0")
bds_x = [Tt.Bond(Tt.BD_IN,5),Tt.Bond(Tt.BD_OUT,5),Tt.Bond(Tt.BD_OUT,3)]
bds_y = [Tt.Bond(Tt.BD_IN,2),Tt.Bond(Tt.BD_OUT,3)]
x = Tt.UniTensor(bonds=bds_x, labels=[4,3,5],dtype=tor.float64,device=tor.device("cpu"))




y = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,3),Tt.Bond(Tt.BD_OUT,4)])
y.SetElem([1,1,0,1,\
           0,0,0,1,\
           1,1,0,0])
print(y)
u,s,v = Tt.linalg.Svd(y)
print(u)
print(s)
print(v)
print("=========================================")
y = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,3),Tt.Bond(Tt.BD_OUT,4)])
y.SetElem([1,1,0,1,\
           0,0,0,1,\
           1,1,0,0])
print(y)
u,s,v = Tt.linalg.Svd_truncate(y,keepdim=2)
print(u)
print(s)
print(v)

print("=========================================")

x = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,5),Tt.Bond(Tt.BD_OUT,5),Tt.Bond(Tt.BD_OUT,4)], labels=[4,3,5])
y = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,3),Tt.Bond(Tt.BD_OUT,4)],labels=[1,5])

x.Print_diagram()
y.Print_diagram()

c = Tt.Contract(x,y)
c.Print_diagram()

c= Tt.Contract(x,y,inbond_first=False)
c.Print_diagram()
exit(1)

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




