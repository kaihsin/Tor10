import sys
sys.path.append("..")
import Tor10 as Tt
import torch as tor
import copy
import numpy as np 

if len(sys.argv) < 5:
    print(".py <J> <Hx> <chi> <converge>")
    exit(1)

## Params H = [J]SzSz - [Hx]Sx
chi = int(sys.argv[3])
Hx  = float(sys.argv[2])
J   = float(sys.argv[1])
CvgCrit = float(sys.argv[4])

## check:
if(chi<1):
    raise ValueError("[ERROR] bond dimension should be >=1")
if(CvgCrit<=0):
    raise ValueError("[ERROR] converge should be >0")


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

## Create NN terms
Sx = Sx*Hx

TFterm = Tt.Otimes(Sx,I) + Tt.Otimes(I,Sx)
ZZterm = Tt.Otimes(Sz,Sz)
del Sz,Sx,I

H = TFterm + ZZterm
del TFterm,ZZterm

H.Reshape([4,4],new_labels=[0,1],N_inbond=1)
## Create Evov Op.
eH = Tt.ExpH(H*-0.1)
eH.Reshape([2,2,2,2],new_labels=[0,1,2,3],N_inbond=2)
H.Reshape([2,2,2,2],new_labels=[0,1,2,3],N_inbond=2) # this is estimator.


## Create MPS:
#
#     |    |     
#   --A-la-B-lb-- 
#
A = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,chi),Tt.Bond(Tt.BD_OUT,2),Tt.Bond(Tt.BD_OUT,chi)],
                 labels=[-1,0,-2]).Rand()
B = Tt.UniTensor(bonds=A.bonds,labels=[-3,1,-4]).Rand()

la = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,chi),Tt.Bond(Tt.BD_OUT,chi)],
              labels=[-2,-3],is_diag=True).Rand()
lb = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,chi),Tt.Bond(Tt.BD_OUT,chi)],
              labels=[-4,-5],is_diag=True).Rand()


## Evov:
Elast = 0
for i in range(100000):
    A.SetLabels([-1,0,-2])
    B.SetLabels([-3,1,-4])
    la.SetLabels([-2,-3])
    lb.SetLabels([-4,-5])

    X = Tt.Contract(Tt.Contract(A,la,inbond_first=False),Tt.Contract(B,lb,inbond_first=False))
    lb.SetLabel(-1,idx=1)
    X = Tt.Contract(lb,X)

    ## X =
    #           (0)  (1)
    #            |    |     
    #  (-4) --lb-A-la-B-lb-- (-5) 
    #
    #X.Print_diagram()
    XNorm = Tt.Contract(X, X,inbond_first=False)
    XH = Tt.Contract(X, H,inbond_first=False)
    #XH.Print_diagram()
    XH.SetLabels([-4,-5,0,1]) ## JJ, this is your bug.
    XHX = Tt.Contract(X, XH,inbond_first=False)
    XeH = Tt.Contract(X,eH,inbond_first=False)
    
    # measurements
    E = (XHX.Storage / XNorm.Storage).item()
    if np.abs(E - Elast) < CvgCrit:
        print("[Converged!]")
        break

    Elast = E
    print("Energy = {:.6f}".format(E))

    XeH.Permute([-4,2,3,-5],N_inbond=2,by_label=True)
    XeH.Contiguous()
    XeH.Reshape([chi*2,chi*2],N_inbond=1)

    A,la,B = Tt.Svd_truncate(XeH,chi)

    la *= 1./la.Norm()

    A.Reshape([chi,2,chi], new_labels=[-1,0,-2], N_inbond=1)
    B.Reshape([chi,2,chi], new_labels=[-3,1,-4], N_inbond=1)

    # de-contract the lb tensor , so it returns to 
    #             
    #            |     |     
	#       --lb-A'-la-B'-lb-- 
	#
	# again, but A' and B' are updated 
	

    lb_inv = Tt.Inverse(lb)
    A = Tt.Contract(lb_inv, A,inbond_first=False)
    B = Tt.Contract(B, lb_inv,inbond_first=False)


    # translation symmetry, exchange A and B site
    A,B = B,A
    la,lb = lb,la 

del X
## Trainsion save:
#Tt.Save(A,"A")
#Tt.Save(B,"B")  
#Tt.Save(la,"la")
#Tt.Save(lb,"lb")
print("[Done]")




