import Tor10 as Tt

import numpy as np 
import torch as tor
import copy


## Single Qnum:
bd_sym_1 = Tt.Bond(Tt.BD_IN,3,qnums=[[0],[1],[2]])
bd_sym_2 = Tt.Bond(Tt.BD_IN,4,qnums=[[-1],[2],[0],[2]])
bd_sym_3 = Tt.Bond(Tt.BD_OUT,5,qnums=[[4],[2],[-1],[5],[1]])
sym_T = Tt.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],labels=[10,11,12],dtype=tor.float64)
sym_T.Print_diagram()
print(sym_T.GetBlock(2))


block_qnum_2 = np.arange(3).reshape(3,1)
sym_T.PutBlock(block_qnum_2,2)
print(sym_T)
print(sym_T.GetBlock(2))


## multiple Qnum:
## U1 x U1 x Z2 x Z4
## U1 = {-2,-1,0,1,2}
## Z2 = {-1,1}
## Z4 = {0,1,2,3}

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
tqin, tqout = sym_T.GetTotalQnums()
print(tqin,tqout)
print(sym_T.GetBlock(1,1,-2,3))



## Test contract.
sym_T2 = Tt.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],labels=[0,2,4],dtype=tor.float64)
Tout = Tt.Contract(sym_T,sym_T2)
Tout.Print_diagram()


## Test mismatch:
#sym_T3 =  Tt.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],labels=[2,0,4],dtype=tor.float64)
#Terr = Tt.Contract(sym_T,sym_T3)


exit(1)




