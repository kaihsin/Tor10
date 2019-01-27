import Tor10 as Tt

import numpy as np 
import torch as tor
import copy

ntwrk = Tt.Network()
ntwrk.Fromfile("test.net")
print(ntwrk)

A = Tt.UniTensor([Tt.Bond(Tt.BD_IN,3),Tt.Bond(Tt.BD_IN,4),Tt.Bond(Tt.BD_OUT,3),Tt.Bond(Tt.BD_OUT,4)]).Rand()
B = Tt.UniTensor([Tt.Bond(Tt.BD_OUT,3),Tt.Bond(Tt.BD_OUT,2)]).Rand()
C = Tt.UniTensor([Tt.Bond(Tt.BD_OUT,4),Tt.Bond(Tt.BD_OUT,4)]).Rand()


ntwrk.Put("A",A)
print(ntwrk)
ntwrk.Put("C",C)
print(ntwrk)
ntwrk.Put("B",B)

TOUT = ntwrk.Launch()

TOUT.Print_diagram()

