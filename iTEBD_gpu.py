import sys
sys.path.append("..")
import tor10 as Tt
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
if chi<1:
    raise ValueError("[ERROR] bond dimension should be >=1")
if CvgCrit<=0:
    raise ValueError("[ERROR] converge should be >0")


## Create onsite-Op.
Sz = Tt.UniTensor(bonds=[Tt.Bond(2),Tt.Bond(2)],rowrank=1,dtype=tor.float64,device=tor.device("cpu"))
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

H = H.Reshape([4,4],new_labels=[0,1],rowrank=1)
## Create Evov Op.
eH = Tt.ExpH(H*-0.1)
eH = eH.Reshape([2,2,2,2],new_labels=[0,1,2,3],rowrank=2)
H = H.Reshape([2,2,2,2],new_labels=[0,1,2,3],rowrank=2) # this is estimator.


## Create MPS:
#
#     |    |     
#   --A-la-B-lb-- 
#
A = Tt.UniTensor(bonds=[Tt.Bond(chi),Tt.Bond(2),Tt.Bond(chi)],
                 rowrank=1,
                 labels=[-1,0,-2]).Rand()
B = Tt.UniTensor(bonds=A.bonds,rowrank=1,labels=[-3,1,-4]).Rand()

la = Tt.UniTensor(bonds=[Tt.Bond(chi),Tt.Bond(chi)],
                rowrank=1,
              labels=[-2,-3],is_diag=True).Rand()
lb = Tt.UniTensor(bonds=[Tt.Bond(chi),Tt.Bond(chi)],
              rowrank=1,
              labels=[-4,-5],is_diag=True).Rand()


H.to(tor.device("cuda:0"))
eH.to(tor.device("cuda:0"))
A.to(tor.device("cuda:0"))
B.to(tor.device("cuda:0"))
la.to(tor.device("cuda:0"))
lb.to(tor.device("cuda:0"))

## Evov:
Elast = 0
for i in range(100000):
    A.SetLabels([-1,0,-2])
    B.SetLabels([-3,1,-4])
    la.SetLabels([-2,-3])
    lb.SetLabels([-4,-5])

    X = Tt.Contract(Tt.Contract(A,la),Tt.Contract(B,lb))
    lb.SetLabel(-1,idx=1)
    X = Tt.Contract(lb,X)

    ## X =
    #           (0)  (1)
    #            |    |     
    #  (-4) --lb-A-la-B-lb-- (-5) 
    #
    #X.Print_diagram()
    Xt = copy.deepcopy(X)
    #Xt.Whole_transpose()
    XNorm = Tt.Contract(X, Xt)
    XH = Tt.Contract(X, H)
    XH.SetLabels([-4,-5,0,1]) ## JJ, this is your bug.
    XHX = Tt.Contract(Xt, XH)
    XeH = Tt.Contract(X,eH)
    
    # measurements
    E = (XHX.Storage / XNorm.Storage).item()
    if np.abs(E - Elast) < CvgCrit:
        print("[Converged!]")
        break

    Elast = E
    print("Energy = {:.6f}".format(E))
    #XeH.Print_diagram()
    XeH.Permute([-4,2,3,-5],by_label=True)
    #XeH.Contiguous_()
    #XeH.View_([chi*2,chi*2],rowrank=1)

    XeH = XeH.Reshape([chi*2,chi*2],rowrank=1)

    A,la,B = Tt.Svd_truncate(XeH,chi)

    la *= la.Norm()**-1

    A = A.Reshape([chi,2,chi], new_labels=[-1,0,-2], rowrank=1)
    B = B.Reshape([chi,2,chi], new_labels=[-3,1,-4], rowrank=1)

    # de-contract the lb tensor , so it returns to 
    #             
    #            |     |     
	#       --lb-A'-la-B'-lb-- 
	#
	# again, but A' and B' are updated 
	

    lb_inv = Tt.Inverse(lb)
    A = Tt.Contract(lb_inv, A)
    B = Tt.Contract(B, lb_inv)


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




