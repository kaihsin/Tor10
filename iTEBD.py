import Tor10 as Tt
import torch as tor
import copy
import numpy as np 


## Params H = [J]SzSz - [Hx]Sx
chi = 4
Hx  = 2.5
J   = -1.



## Create onsite-Op.
Sz = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,2),Tt.Bond(Tt.BD_OUT,2)],dtype=tor.float64,device=tor.device("cpu"))
Sx = copy.deepcopy(Sz)
I  = copy.deepcopy(Sz)
Sz.SetElem([1, 0,\
            0,-1 ])
Sx.SetElem([0, 1,\
            1, 0 ])
I.SetElem([1, 0,\
           0, 1 ])
Sz = Sz*J
Sz2 = copy.deepcopy(Sz)

## Create NN terms
Sx.SetLabels([2,3]) 
Sx = Sx*Hx
I.SetLabels([4,5])
print(Sx)
print(I)
TFterm = Tt.Contract(Sx,I) + Tt.Contract(I,Sx)
del Sx, I

Sz2.SetLabels([2,3])
ZZterm = Tt.Contract(Sz,Sz2)
del Sz,Sz2

H = TFterm + ZZterm
del TFterm,ZZterm

H.Contiguous()
H.Reshape([4,4],new_labels=[0,1],N_inbond=1)

## Create Evov Op.
eH = Tt.ExpH(H)
eH.Reshape([2,2,2,2],new_labels=[0,1,2,3],N_inbond=2)
H.Reshape([2,2,2,2],new_labels=[0,1,2,3],N_inbond=2) # this is estimator.


## Create MPS:
#
#     |    |     
#   --A-la-B-lb-- 
#
A = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,chi),Tt.Bond(Tt.BD_OUT,2),Tt.Bond(Tt.BD_OUT,chi)],
              labels=[-1,0,-2])
B = Tt.UniTensor(bonds=A.bonds,labels=[-3,1,-4])
A.Rand()
B.Rand()

la =  Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,chi),Tt.Bond(Tt.BD_OUT,chi)],
              labels=[-2,-3],is_diag=True)
lb = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,chi),Tt.Bond(Tt.BD_OUT,chi)],
              labels=[-4,-5],is_diag=True)


## Evov:
 
X = Tt.Contract(Tt.Contract(A,la),Tt.Contract(B,lb))
X = Tt.Contract(X,eH)
lb.SetLabel(-1,idx=1)
X = Tt.Contract(lb,X)
X.Permute([-4,2,3,-5],N_inbond=2,by_label=True)
X.Contiguous()
X.Reshape([chi*2,chi*2],N_inbond=1)
A,la,B = Tt.Svd(X)
del X





