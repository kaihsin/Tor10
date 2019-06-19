import torch 
import copy,os
import numpy as np
import pickle as pkl
import re
from .UniTensor import *

class Network:
    def __init__(self,nwfile=None,delimiter=None):
        """
        Constructor of the Network

        The Network is an object that allow one to create a complex network from a pre-defined Network file. By putting the Tensors into the Network, the user simply call "Network.Launch()" to get the out-come.

        Args:
            nwfile:
                The path of the Network file. 

            delimiter:
                The delimiter that is used to parse the Network file.

        Example:
            The following is an simple example for a Network file ("test.net").
            ::
                A : -1 -2; 1 2 
                B : 1;  3
                C : 2;  4
                TOUT: -1 -2; 3 4
                Order: (A,B),C

            
            * Each line defines a Tensor with the left side of the colon is the name of the tensor. The right side of the colon defines the labels of each bonds. 
        
            * The semicolon seperates the row-space and col-space, it is equivalent as "rowrank" property of the UniTensor. The left side of semicolon is defined as row-space, and right side of semicolon is degined as col-space.
        
            [Note] that there are two preserved tensor name "TOUT" and "Order". The "TOUT" specify the output tensor, and "Order" defineds how the Tensors in the Network will be contracted. 

            [Note] The "Order" is not required. If not specify, the tensors will contract one by one accroding to the sequence as they appears in the Network file.   
        
            * The above Network file means:

            1. an "A" UniTensor with 2 inbonds label [-1,-2], 2 outbonds label [1,2]

            2. an "B" UniTensor with 1 inbonds, 1 outbonds label [1,3]

            3. an "C" UniTensor with 1 inbonds, 1 outbonds label [2,4]
    
            4. the output UniTensor will be 2 in-bonds with labels [-1,-2] and 2 out-bonds with labels [3,4].

            5. the UniTensors in the Network will contract accroding to: "A" and "B" contract first, then contract with C to form the final output UniTensor.


        >>> Ntwk = tor10.Network("test.net",delimiter=",")
        >>> print(Ntwk)
        ==== Network ====
        [x] A : -1 -2 ; 1 2 
        [x] B : ; 1 3 
        [x] C : ; 2 4 
        TOUT : -1 -2 ; 3 4 
        =================



        """        
        self.tensors = None # This is the nested list of new labels
        self.TOUT = None    # This is nested list for the labels of out label
        if nwfile is not None:
            self.fromfile(nwfile,delimiter)

        self.instances = None
        self.Order = None
        
    def Fromfile(self,ntwrk_file,delimiter=None):
        """
        Read the Network file.
        
        Args:
            nwfile:
                The path of the Network file. 

            delimiter:
                The delimiter that is used to parse the Network file.

        """
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
                raise TypeError("Network.fromfile","[ERROR] The network file have wrong format at line [%d]" % i)
            Name = tmp[0].strip()
            if Name == 'Order':
                if self.Order is not None:
                    raise TypeError("Network.fromfile","[ERROR] The network file have multiple line of [Order]")
                tmp = tmp[1].strip()

                ## check if the parenthese are matching.
                if not self.__is_matched(tmp):
                    raise TypeError("Network.fromfile","[ERROR] The parentheses mismatch happend for the [Order] at line [%d]" % i)
                
                self.Order = tmp                
                
            else:
                tmp = tmp[1].split(';')
                if(len(tmp)!=2):
                    if Name != 'TOUT':
                        raise Exception("Network.fromfile","[ERROR] syntax error for Network file.");
                    Inbonds, Outbonds = None,None
                else:
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
                    print("%d " % i, end="")
                print("; ",end="")
                for i in val[1]:
                    print("%d " % i, end="")
                print("")

            print("TOUT : ",end="")
            for i in self.TOUT[0]:
                print("%d " % i, end="")
            print("; ",end="")
            for i in self.TOUT[1]:
                print("%d " % i, end="")
            print("")
        print("=================")
    
    ## private function
    def __is_matched(self,expression):
        queue = []

        for letter in expression:
            if letter == '(':
                queue.append(')')
            elif letter == ')':
                if not queue or letter != queue.pop():
                    return False
        return not queue

    def Put(self,name,tensor):
        """
        Put the UniTensor into the tensor named [name] in the Network. To use the Network, only untagged tensor can be put into the Network.

        Args:
            name: 
                The name of the tensor defines in the Network to put the UniTensor.

            tensor:
                A UniTensor that is to be put into the Network. It should be untagged.

            
        Example:
        ::
            ntwrk = tor10.Network()
            ntwrk.Fromfile("test.net")
            A = tor10.UniTensor([tor10.Bond(3),tor10.Bond(4),tor10.Bond(3),tor10.Bond(4)],rowrank=2)
            B = tor10.UniTensor([tor10.Bond(3),tor10.Bond(2)],rowrank=1)
            C = tor10.UniTensor([tor10.Bond(4),tor10.Bond(4)],rowrank=1)


        >>> print(ntwrk)
        ==== Network ====
        [x] A : -1 -2 ; 1 2 
        [x] B : ; 1 3 
        [x] C : ; 2 4 
        TOUT : -1 -2 ; 3 4 
        =================

        
        >>> ntwrk.Put("A",A)
        >>> print(ntwrk)
        ==== Network ====
        [o] A : -1 -2 ; 1 2 
        [x] B : ; 1 3 
        [x] C : ; 2 4 
        TOUT : -1 -2 ; 3 4 
        =================


        >>> ntwrk.Put("C",C)
        >>> print(ntwrk)
        ==== Network ====
        [o] A : -1 -2 ; 1 2 
        [x] B : ; 1 3 
        [o] C : ; 2 4 
        TOUT : -1 -2 ; 3 4 
        =================


        >>> ntwrk.Put("B",B)
        >>> print(ntwrk)
        ==== Network ====
        [o] A : -1 -2 ; 1 2 
        [o] B : ; 1 3 
        [o] C : ; 2 4 
        TOUT : -1 -2 ; 3 4 
        =================


        >>> TOUT = ntwrk.Launch()
        >>> TOUT.Print_diagram(bond_info=True)
        -----------------------
        tensor Name : 
        tensor Rank : 4
        has_symmetry: False
        on device     : cpu
        is_diag       : False
                    -------------      
                   /             \     
            -1 ____| 3         2 |____ 3  
                   |             |     
            -2 ____| 4         4 |____ 4  
                   \             /     
                    -------------      
        lbl:-1 Dim = 3 |
        REG     :
        _
        lbl:-2 Dim = 4 |
        REG     :
        _
        lbl:3 Dim = 2 |
        REG     :
        _
        lbl:4 Dim = 4 |
        REG     :


        """
        ## check if the Network is set.
        if self.tensors is None:
            raise ValueError("Network.put","[ERROR] Network hasn't been constructed. Construct a Netwrok before put the tensors.")

       
        ## checking tensor is UniTensor
        if not isinstance(tensor,UniTensor):
            raise TypeError("Network.put","[ERROR] Network can only accept UniTensor")

        ## tmp check blockform:
        if tensor.braket is not None:
            raise TypeError("Network.put","[ERROR] currently Network can only accept untagged UniTensor") 

        ## check if the name in the Network
        ## remaining thing: how to deal with in, out bond?
        if name in self.tensors:
            ##checking:
            if len(tensor.shape) != len(self.tensors[name][0]) + len(self.tensors[name][1]):
                raise TypeError("Network.put","[ERROR] Trying to put tensor %s that has different rank" % name)
            if self.instances is None:
                self.instances = {name:tensor}
            else:
                self.instances[name] = tensor
        else:
            raise ValueError("Network.put","[ERROR] Network does not contain the tensor with name [%s]" % name)
    
    
    def __launch_by_order(self):
        
        # [Developer note]
        # This is using a variance of of Shunting-yard Algo. to evaluate the "Order" string and contract accrodinly.
        


        ## lambda function
        peek = lambda stack: stack[-1] if stack else None

        tokens = re.findall("[(,)]|\w+", self.Order) 
        ##[Note] self.Order exists should be check before calling this function
        values = []

        old_labels = dict()
        for key in self.tensors.keys():
            old_labels[key] = copy.copy(self.instances[key].labels)
            self.instances[key].labels =  np.array(self.tensors[key][0].tolist() + self.tensors[key][1].tolist())
        

        operators = []
        first = True
        for token in tokens:
        
            if token == '(':
                operators.append(token)
            elif token == ')':
                top = peek(operators)
                while top is not None and top != '(':
                    operators.pop()
                    # apply contract                        
                    left = values.pop()
                    right = values.pop()
                    values.append(Contract(left,right))
                    top = peek(operators)
                operators.pop() # Discard the '('
            elif token == ',':
                # Operator
                top = peek(operators)
                while top is not None and top not in "()":
                    operators.pop()
                    # apply contract
                    left = values.pop()
                    right = values.pop()
                    values.append(Contract(left,right))
                    
                    top = peek(operators)
                operators.append(token)
            elif len(re.findall("\W+",token)) > 0:
                raise ValueError("String Contain invalid symbol [%s]" % token)
            else:
                #print(token)
                values.append(self.instances[token])

        while peek(operators) is not None:
            operators.pop()
            # apply contract
            left = values.pop()
            right = values.pop()
            values.append(Contract(left,right))

        for key in self.tensors.keys():
            self.instances[key].labels = old_labels[key]

 
        return values[0]

    def Launch(self):
        """
        Launch the Network. When call, it check if there is a well defined Network structure (Network file loaded), if there is "TOUT" tensor for the Network and if all the Tensors the is being defined in the strucutre are set. 

        Return:
            UniTensor, the shape and the bonds property are the same as defined in "TOUT"

        """
        if self.tensors is None:
            raise Exception("Network","[ERROR] No in-put tensors for the Network")
        if self.TOUT is None:
            raise Exception("Network","[ERROR] No TOUT tensor for the Network")
        if self.instances is None:
            raise Exception("Network","[ERROR] No UniTensor is put in the Network")

        for key in self.tensors.keys():
            if not key in self.instances:
                raise Exception("Network","[ERROR] The [%s] tensor is not put in the Network" % key)

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
            out = self.__launch_by_order()            

        if(len(out.shape)==0):
            return out
        else:
            per_lbl = self.TOUT[0].tolist() + self.TOUT[1].tolist()
            out.Permute(per_lbl,len(self.TOUT[0]),by_label=True)
            ## this is temporary, not finished!!!
            #print("Network.Launch is currently under developing.")

            return out
    
