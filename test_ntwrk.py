import Tor10 as Tt

import numpy as np 
import torch as tor
import copy

ntwrk = Tt.Network()
ntwrk.Fromfile("test.net")
print(ntwrk)

A = Tt.UniTensor([Tt.Bond(3),Tt.Bond(4),Tt.Bond(3),Tt.Bond(4)],2).Rand()
B = Tt.UniTensor([Tt.Bond(3),Tt.Bond(2)],0).Rand()
C = Tt.UniTensor([Tt.Bond(4),Tt.Bond(4)],0).Rand()

ntwrk.Put("A",A)
print(ntwrk)
ntwrk.Put("C",C)
print(ntwrk)
ntwrk.Put("B",B)

TOUT = ntwrk.Launch()

TOUT.Print_diagram()

