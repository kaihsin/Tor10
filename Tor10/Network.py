import torch 
import copy,os
import numpy as np
import pickle as pkl
import re
from .Tensor import *

class Network():
    def __init__(self,nwfile=None,delimiter=None):
        self.tensors = None
        self.TOUT = None
        if nwfile is not None:
            self.fromfile(nwfile,delimiter)

        self.instances = None
        self.Order = None
        
    def Fromfile(self,ntwrk_file,delimiter=None):
        self.tensors = None
        self.TOUT = None
        self.instances = None
        self.Order = None

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
            if Name == 'Order':
                if self.Order is not None:
                    raise TypeError("Network.fromfile","[ERROR] The network file have multiple line of [Order]")
                tmp = tmp[1].strip()

                ## check if the parenthese are matching.
                if not self.__is_matched(tmp):
                    raise TypeError("Network.fromfile","[ERROR] The parentheses mismatch happend for the [Order] at line [%d]"%(i))
                
                self.Order = tmp                
                
            else:
                tmp = tmp[1].split(';')
                if delimiter is None:
                    Inbonds = np.array(tmp[0].strip().split(),dtype=np.int)
                    Outbonds= np.array(tmp[1].strip().split(),dtype=np.int)
                else:
                    Inbonds = np.array(tmp[0].strip().split(delimiter),dtype=np.int)
                    Outbonds= np.array(tmp[1].strip().split(delimiter),dtype=np.int)
                tn_shell = [Inbonds,Outbonds]
                if Name == 'TOUT':
                    if self.TOUT is not None:
                        raise TypeError("Network.fromfile","[ERROR] The network file have multiple line of [TOUT]")

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

        #print(self.tensors)
        #print(self.TOUT)


    def __repr__(self):
        self.__draw()
        return ""
    def __str__(self):
        self.__draw()
        return ""

    ## This is the private function 
    def __draw(self):
        print("==== Network ====")
        if self.tensors is None:
            print("      Empty      ")
        else:
            for key, val in self.tensors.items():
                status = "x"
                if self.instances is not None:
                    if key in self.instances:
                        status = "o"

                print("[%s] %s : "%(status,key),end="")
                for i in val[0]:
                    print("%d "%(i),end="")
                print("; ",end="")
                for i in val[1]:
                    print("%d "%(i),end="")
                print("")

            print("TOUT : ",end="")
            for i in self.TOUT[0]:
                print("%d "%(i),end="")
            print("; ",end="")
            for i in self.TOUT[1]:
                print("%d "%(i),end="")
            print("")
        print("=================")
    
    def __is_matched(expression):
        queue = []

        for letter in expression:
            if letter == '(':
                queue.append(')')
            elif letter == ')':
                if not queue or letter != queue.pop():
                    return False
        return not queue

    def Put(self,name,tensor):
        ## check if the Network is set.
        if self.tensors is None:
            raise ValueError("Network.put","[ERROR] Network hasn't been constructed. Construct a Netwrok before put the tensors.")

        ## checking tensor is UniTensor
        if not isinstance(tensor,UniTensor):
            raise TypeError("Network.put","[ERROR] Network can only accept UniTensor")

        ## check if the name in the Network
        ## remaining thing: how to deal with in, out bond?
        if name in self.tensors:
            ##checking:
            if len(tensor.shape()) != len(self.tensors[name][0]) + len(self.tensors[name][1]):
                raise TypeError("Network.put","[ERROR] Trying to put tensor %s that has different rank"%(name))
            if self.instances is None:
                self.instances = {name:tensor}
            else:
                self.instances[name] = tensor
        else:
            raise ValueError("Network.put","[ERROR] Network does not contain the tensor with name [%s]"%(name)) 
    


    def Launch(self):
        if self.tensors is None:
            raise Exception("Network","[ERROR] No in-put tensors for the Network")
        if self.TOUT is None:
            raise Exception("Network","[ERROR] No TOUT tensor for the Network")
        if self.instances is None:
            raise Exception("Network","[ERROR] No UniTensor is put in the Network")

        for key in self.tensors.keys():
            if not key in self.instances:
                raise Exception("Network","[ERROR] The [%s] tensor is not put in the Network"%(key))

        out = None
        if self.Order is None:
            for key,value in self.instances.items():
                if out is None:
                    
                    out = copy.deepcopy(value)
                    out.labels = np.array(self.tensors[key][0].tolist() + self.tensors[key][1].tolist())
                else:
                    old_labels = copy.copy(value.labels)
                    value.labels = np.array(self.tensors[key][0].tolist() + self.tensors[key][1].tolist())
                    out = Contract(out,value)
                    value.labels = old_labels
        else :
            ##Unfin
            print("Order is under developing")
            exit(1)
    
        per_lbl = self.TOUT[0].tolist() + self.TOUT[1].tolist()
        out.Permute(per_lbl,len(self.TOUT[0]),by_label=True)
        ## this is temporary, not finished!!!
        #print("Network.Launch is currently under developing.")



        return out
    
