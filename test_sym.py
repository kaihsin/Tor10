import Tor10 as Tt

import numpy as np 
import torch as tor
import copy


## Single Qnum:
"""
bd_sym_1 = Tt.Bond(Tt.BD_IN,3,qnums=[[0],[1],[2]])
bd_sym_2 = Tt.Bond(Tt.BD_IN,4,qnums=[[-1],[2],[0],[2]])
bd_sym_3 = Tt.Bond(Tt.BD_OUT,5,qnums=[[4],[2],[-1],[5],[1]])
sym_T = Tt.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],labels=[10,11,12],dtype=tor.float64)
sym_T.Print_diagram()
q_in,q_out = sym_T.GetTotalQnums()
print(q_in)
print(q_out)
print(sym_T.GetBlock(2))
exit(1)

block_qnum_2 = np.arange(3).reshape(3,1)
sym_T.PutBlock(block_qnum_2,2)
print(sym_T)
print(sym_T.GetBlock(2))
"""
bd_out_mulsym = Tt.Bond(Tt.BD_OUT,3,qnums=[[-2,0,0],\
                                                 [-1,1,3],\
                                                 [ 1,0,2]],\
                                          sym_types=[Tt.Symmetry.U1(),\
                                                     Tt.Symmetry.Zn(2),\
                                                     Tt.Symmetry.Zn(4)])
print(bd_out_mulsym)

c = Tt.Bond(Tt.BD_OUT,2,qnums=[[0,1,-1],[1,1,0]])
d = Tt.Bond(Tt.BD_OUT,2,qnums=[[1,0,-1],[1,0,0]])
c.combine(d)
print(c)

## multiple Qnum:
## U1 x U1 x U1 x U1
## U1 = {-2,-1,0,1,2}
## U1 = {-1,1}
## U1 = {-1,1,2,3}
bd_sym_1 = Tt.Bond(Tt.BD_IN,3,qnums=[[0, 2, 1, 0],\
                                     [1, 1,-1, 1],\
                                     [2,-1, 1, 0]])

bd_sym_2 = Tt.Bond(Tt.BD_IN,4,qnums=[[-1, 0,-1, 3],\
                                     [ 0, 0,-1, 2],\
                                     [ 1, 0, 1, 0],\
                                     [ 2,-2,-1, 1]])

bd_sym_3 = Tt.Bond(Tt.BD_OUT,2,qnums=[[-1,-2,-1,2],\
                                      [ 1, 1, -2,3]])

sym_T = Tt.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],labels=[1,2,3],dtype=tor.float64)

#print("1")
#print(bd_sym_1)
#print("2")
#print(bd_sym_2)

tqin, tqout = sym_T.GetTotalQnums()
print(tqin,tqout)
#exit(1)
print(sym_T.GetBlock(1,1,-2,3))
exit(1)
sym_T.Print_diagram()

## Test contract.
sym_T2 = Tt.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],labels=[0,2,4],dtype=tor.float64)
sym_T2.Print_diagram()
Tout = Tt.Contract(sym_T,sym_T2)
Tout.Print_diagram()

"""
## Test mismatch:
#sym_T3 =  Tt.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],labels=[2,0,4],dtype=tor.float64)
#Terr = Tt.Contract(sym_T,sym_T3)


exit(1)
"""



