import torch 
import copy,os
import numpy as np
import pickle as pkl
from .Tensor import *

class Network():
    def __init__(self,nwfile=None,delimiter=None):
        self.tensors = None
        self.TOUT = None
        if nwfile is not None:
            self.fromfile(nwfile,delimiter)

    def fromfile(self,ntwrk_file,delimiter=None):
        self.tensors = None
        self.TOUT = None

        f = open(ntwrk_file,'r')
        lines = f.readlines()
        f.close()
        for i in range(len(lines)):
            line = lines[i]
            line = line.strip()
            ## check empty line
            if line == '':
                continue

            ## decomp :
            tmp = line.split(':')
            if len(tmp) != 2:
                raise TypeError("Network.fromfile","[ERROR] The network file have wrong format at line [%d]"%(i))
            Name = tmp[0].strip()
            tmp = tmp[1].split(';')
            if delimiter is None:
                Inbonds = tmp[0].strip().split()
                Outbonds= tmp[1].strip().split()
            else:
                Inbonds = tmp[0].strip().split(delimiter)
                Outbonds= tmp[1].strip().split(delimiter)
            tn_shell = [Inbonds,Outbonds]
            if Name == 'TOUT':
                self.TOUT = tn_shell
            else:
                if self.tensors is None:
                    self.tensors = {Name:tn_shell}
                else:
                    if Name in self.tensors:
                        raise ValueError("Network.fromfile","[ERROR] network file contain duplicate tensor names [%s] at line [%d]"%(Name,i))
                    self.tensors[Name] = tn_shell
            
        ## final checking:
        if self.TOUT is None:
            raise TypeError("Network.fromfile","[ERROR] network file have no TOUT element.")
            
        if self.tensors is None:
            raise TypeError("Network.fromfile","[ERROR] network file have no input elements exist.")

        print(self.tensors)
        print(self.TOUT)
    def draw():
        pass 



