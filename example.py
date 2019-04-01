import Tor10 
import numpy as np 
import torch 
import copy

## Example for Tor10 v0.2
## Kai-Hsin Wu


## Bond:
#=======================================
## Non-symmetry:
bd_x = Tor10.Bond(3) ## This is equivalent to Tor10.Bond(3,Tor10.BD_REGULAR)
bd_y = Tor10.Bond(4)
bd_z = Tor10.Bond(3)
print(bd_x)
print(bd_y)
print(bd_x==bd_z) ## This should be true
print(bd_x is bd_z) ## This should be false
print(bd_x==bd_y) ## This should be false

## Symmetry:
#> U1
bd_sym_U1 = Tor10.Bond(3,qnums=[[-1],[0],[1]])
print(bd_sym_U1)

#> Z2 
bd_sym_Z2 = Tor10.Bond(3,qnums=[[0],[1],[0]],sym_types=[Tor10.Symmetry.Zn(2)])
print(bd_sym_Z2)

#> Z4
bd_sym_Z4 = Tor10.Bond(3,qnums=[[0],[2],[3]],sym_types=[Tor10.Symmetry.Zn(4)])
print(bd_sym_Z4)

#> Multiple U1
bd_sym_multU1 = Tor10.Bond(3,qnums=[[-2,-1,0,-1],
                                    [1 ,-4,0, 0],
                                    [-8,-3,1, 5]])
print(bd_sym_multU1)

#> Multiple mix symmetry: U1 x Z2 x Z4
bd_sym_mix = Tor10.Bond(3,qnums=[[-2,0,0],
                                 [-1,1,3],
                                 [ 1,0,2]],
                         sym_types=[Tor10.Symmetry.U1(),
                                    Tor10.Symmetry.Zn(2),
                                    Tor10.Symmetry.Zn(4)]) 
print(bd_sym_mix)


## Combine:
a = Tor10.Bond(3)
b = Tor10.Bond(4)
c = Tor10.Bond(2,qnums=[[0,1,-1],[1,1,0]])
d = Tor10.Bond(2,qnums=[[1,0,-1],[1,0,0]]) 
e = Tor10.Bond(2,qnums=[[1,0],[1,0]])
a.combine(b)
print(a)

c.combine(d)
print(c)

## UniTensor:
#=========================================================
# Create Tensor
a = Tor10.UniTensor(bonds=[Tor10.Bond(3),Tor10.Bond(4)],N_inbond=1)
print(a)
c = Tor10.UniTensor(bonds=[Tor10.Bond(3),Tor10.Bond(4),Tor10.Bond(5)],N_inbond=1,labels=[-3,4,1])
print(c)
## Execute this only when CUDA is installed along with pytorch 
#d = Tor10.UniTensor(bonds=[Tor10.Bond(3),Tor10.Bond(4)],N_inbond=1,device=torch.device("cuda:0"))
e = Tor10.UniTensor(bonds=[Tor10.Bond(6),Tor10.Bond(6)],N_inbond=1,is_diag=True)
f = Tor10.UniTensor(bonds=[Tor10.Bond(3),Tor10.Bond(4),Tor10.Bond(5)],N_inbond=2,labels=[-3,4,1],dtype=torch.float32)

print(e.shape())

# Labels related 
g = Tor10.UniTensor(bonds=[Tor10.Bond(3),Tor10.Bond(4)],N_inbond=1,labels=[5,6])
print(g.labels)
g.SetLabel(-1,1)
print(g.labels)

new_label=[-1,-2]
g.SetLabels(new_label)
print(g.labels)

# Element related
Sz = Tor10.UniTensor(bonds=[Tor10.Bond(2),Tor10.Bond(2)],N_inbond=1,
                              dtype=torch.float64,
                              device=torch.device("cpu"))
Sz.SetElem([1, 0,
            0,-1])
print(Sz)

# Sparse
a = Tor10.UniTensor(bonds=[Tor10.Bond(3),Tor10.Bond(3)],N_inbond=1,is_diag=True)
print(a)

a.Todense()
print(a)

# CombineBonds:
bds_x = [Tor10.Bond(5),Tor10.Bond(5),Tor10.Bond(3)]
x = Tor10.UniTensor(bonds=bds_x, N_inbond=1, labels=[4,3,5])
y = Tor10.UniTensor(bonds=bds_x, N_inbond=1, labels=[4,3,5])
z = Tor10.UniTensor(bonds=bds_x, N_inbond=1, labels=[4,3,5])
x.Print_diagram()

x.CombineBonds([4,3])
x.Print_diagram()

y.CombineBonds([4,3],is_inbond=True)
y.Print_diagram()

z.CombineBonds([4,3],is_inbond=True,new_label=8)
z.Print_diagram()


# Contiguous()
bds_x = [Tor10.Bond(5),Tor10.Bond(5),Tor10.Bond(3)]
x = Tor10.UniTensor(bonds=bds_x, N_inbond=1, labels=[4,3,5])
print(x.is_contiguous())
x.Permute([0,2,1],N_inbond=1)
print(x.is_contiguous())
x.Contiguous()
print(x.is_contiguous())


# Permute
bds_x = [Tor10.Bond(6),Tor10.Bond(5),Tor10.Bond(3)]
x = Tor10.UniTensor(bonds=bds_x, N_inbond=1,labels=[4,3,5])
y = Tor10.UniTensor(bonds=bds_x, N_inbond=1,labels=[4,3,5])
x.Print_diagram()

x.Permute([0,2,1],N_inbond=2)
x.Print_diagram()

y.Permute([3,4,5],N_inbond=2,by_label=True)
y.Print_diagram()

# Reshape
bds_x = [Tor10.Bond(6),Tor10.Bond(5),Tor10.Bond(3)]
x = Tor10.UniTensor(bonds=bds_x, N_inbond=1,labels=[4,3,5])
x.Print_diagram()

x.Reshape([2,3,5,3],new_labels=[1,2,3,-1],N_inbond=2)
x.Print_diagram()


## GetTotalQnums
bd_sym_1 = Tor10.Bond(3,qnums=[[0, 2, 1, 0],
                                     [1, 1,-1, 1],
                                     [2,-1, 1, 0]])
bd_sym_2 = Tor10.Bond(4,qnums=[[-1, 0,-1, 3],
                                     [ 0, 0,-1, 2],
                                     [ 1, 0, 1, 0],
                                     [ 2,-2,-1, 1]])
bd_sym_3 = Tor10.Bond(2,qnums=[[-1,-2,-1,2],
                                      [ 1, 1, -2,3]])

sym_T = Tor10.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],labels=[1,2,3],N_inbond=2,dtype=torch.float64)
tqin,tqout=sym_T.GetTotalQnums()
print(tqin)
print(tqout)


sym_T_bf = Tor10.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],labels=[1,2,3],N_inbond=2,is_blockform=True,dtype=torch.float64)


## GetBlock
bd_sym_1 = Tor10.Bond(3,qnums=[[0],[1],[2]])
bd_sym_2 = Tor10.Bond(4,qnums=[[-1],[2],[0],[2]])
bd_sym_3 = Tor10.Bond(5,qnums=[[4],[2],[-1],[5],[1]])
sym_T = Tor10.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],N_inbond=2,labels=[10,11,12],dtype=torch.float64)

sym_T.Print_diagram()

q_in, q_out = sym_T.GetTotalQnums()
print(q_in)
print(q_out)
block_2 = sym_T.GetBlock(2)
print(block_2)

## multiple Qnum:
## U1 x U1 x U1 x U1
## U1 = {-2,-1,0,1,2}
## U1 = {-1,1}
## U1 = {0,1,2,3}
bd_sym_1 = Tor10.Bond(3,qnums=[[0, 2, 1, 0],
                               [1, 1,-1, 1],
                               [2,-1, 1, 0]])
bd_sym_2 = Tor10.Bond(4,qnums=[[-1, 0,-1, 3],
                               [ 0, 0,-1, 2],
                               [ 1, 0, 1, 0],
                               [ 2,-2,-1, 1]])
bd_sym_3 = Tor10.Bond(2,qnums=[[-1,-2,-1,2],
                               [ 1, 1, -2,3]])

sym_T = Tor10.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],N_inbond=2,labels=[1,2,3],dtype=torch.float64)

tqin,tqout=sym_T.GetTotalQnums()
print(tqin)
print(tqout)
block_1123 = sym_T.GetBlock(1,1,-2,3)
print(block_1123)

## Contract:
x = Tor10.UniTensor(bonds=[Tor10.Bond(5),Tor10.Bond(5),Tor10.Bond(4)], N_inbond=2,labels=[4,3,5])
y = Tor10.UniTensor(bonds=[Tor10.Bond(4),Tor10.Bond(3)],N_inbond=1,labels=[5,1])
x.Print_diagram()
y.Print_diagram()
c = Tor10.Contract(x,y)
c.Print_diagram()

## From_torch
x = torch.ones(3,3)
y = Tor10.From_torch(x,N_inbond=1,labels=[4,5])
y.Print_diagram()

x2 = torch.ones(3,4,requires_grad=True)
y2 = Tor10.From_torch(x2,N_inbond=1)
print(y2.requires_grad())

# Network:
#==============================
ntwrk = Tor10.Network()
ntwrk.Fromfile("test.net")
A = Tor10.UniTensor([Tor10.Bond(3),Tor10.Bond(4),Tor10.Bond(3),Tor10.Bond(4)],N_inbond=2)
B = Tor10.UniTensor([Tor10.Bond(3),Tor10.Bond(2)],N_inbond=1)
C = Tor10.UniTensor([Tor10.Bond(4),Tor10.Bond(4)],N_inbond=1)
print(ntwrk)
ntwrk.Put("A",A)
print(ntwrk)
ntwrk.Put("B",B)
print(ntwrk)
ntwrk.Put("C",C)
print(ntwrk)
TOUT = ntwrk.Launch()
TOUT.Print_diagram()

# linalg:
#============================
x = Tor10.From_torch(torch.arange(0.1,2.5,0.1).reshape(2,3,4).to(torch.float64),labels=[6,7,8],N_inbond=1)
x.Print_diagram()
factors, core = Tor10.Hosvd(x,order=[7,6,8],bonds_group=[2,1],by_label=True)
core.Print_diagram()
print(len(factors))
factors[0].Print_diagram()
factors[1].Print_diagram()

rep_x = core
for f in factors:
    rep_x = Tor10.Contract(rep_x,f)
rep_x.Permute([6,7,8],N_inbond=1,by_label=True)
print(rep_x - x)

a = Tor10.UniTensor(bonds=[Tor10.Bond(3),Tor10.Bond(4)],N_inbond=1)
b = Tor10.UniTensor(bonds=[Tor10.Bond(4),Tor10.Bond(5)],N_inbond=1)
c = Tor10.UniTensor(bonds=[Tor10.Bond(5),Tor10.Bond(6)],N_inbond=1)   
d = Tor10.UniTensor(bonds=[Tor10.Bond(6),Tor10.Bond(2)],N_inbond=1)

f = Tor10.Chain_matmul(a,b,c,d)
f.Print_diagram()


