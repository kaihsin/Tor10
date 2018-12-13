import torch 
import copy
import numpy as np

from .Bond import *


class UniTensor():

    def __init__(self,D_IN,D_OUT, label=None, device=torch.device("cpu"),dtype=torch.float64,torch_tensor=None):
        """
            @description: This is the initialization of the UniTensor.
            @param      : D_IN  [require]: The in-bonds , it should be an list with len(list) is the # of in-bond, and each element describe the dimension of each bond.  
                          D_OUT [require]: The out-bonds, it should be an list with len(list) is the # of in-bond, and each element describe the dimension of each bond.
                          label [option ]: The customize label. the number of elements should be the same as the total rank of the tensor, and contain on duplicated elements.
                          device[option ]: This should be a [torch.device]. When provided, the tensor will be put on the device ("cpu", "cuda", "cuda:x" with x is the GPU-id. See torch.device for further information.)
                          dtype [option ]: This should be a [torch.dtype ]. The default type is float with either float32 or float64 which follows the same internal rule of pytorch. For further information, see pytorch documentation. 
                          torch_tensor [private]: This is the internal arguments in current version. It should not be directly use, otherwise may cause inconsistence with Bonds and memory layout. 
                                                  ** Developer **
                                                  > The torch_tensor should have the same rank as len(label), and with each bond dimensions strictly the same as describe as in D_IN, D_OUT.      
        """
        self.D_IN = copy.copy(D_IN)
        self.D_OUT = copy.copy(D_OUT)
            

        
        if label is None:
            self.label = np.arange(len(D_IN)+len(D_OUT))
        else:
            self.label = copy.copy(label)
            ## Checking:
            if not len(self.label) == (len(D_IN) + len(D_OUT)):
                raise Exception("UniTensor.__init__","label size is not consistence with the rank")
            if not len(np.unique(self.label)) == len(self.label):
                raise Exception("UniTensor.__init__","label contain duplicate element.")


        if torch_tensor is None:
            DALL = D_IN + D_OUT
            self.Storage = torch.zeros(tuple(DALL), device=device, dtype = dtype)
            del DALL
        else:
            self.Storage = torch_tensor
    

    ## print layout:
    def Print_diagram(self):
        """
            @Description: This is the beauty print of the tensor diagram. Including the information for the placeing device 
                          1.The left hand side is always the In-bond, and the right hand side is always the Out-bond. 
                          2.The number attach to the out-side of each leg is the Bond-dimension. 
                          3.The number attach to the in-side of each leg is the label. 
                          4.The real memory layout are follow clock-wise from upper-right to upper-left.
                          
                          [ex:] Rank = 4. 
                                torch.Tensor dimension: (1,2,3,6) 
                                D_IN=[1,2], D_OUT=[3,6], label=[0,5,3,11]

                                     -----------
                                1  --| 0     3 |-- 3
                                     |         |
                                2  --| 5    11 |-- 6
                                     -----------

        """
        print("tensor Rank : %d"%(len(self.label)))
        print("on device   : %s"%(self.Storage.device))        
        print("")        
        if len(self.D_IN) > len(self.D_OUT):
            vl = len(self.D_IN)
        else:
            vl = len(self.D_OUT)

        print(vl)
        print("       ---------------     ")
        for i in range(vl):
            print("       |             |     ")
            if(i<len(self.D_IN)):
                l = "%3d--"%(self.D_IN[i])
                llbl = "%-3d"%(self.label[i]) 
            else:
                l = "     "
                llbl = "   "
            if(i<len(self.D_OUT)):
                r = "--%-3d"%(self.D_OUT[i])
                rlbl = "%3d"%(self.label[len(self.D_IN)+i])
            else:
                r = "     "
                rlbl = "   "
            print("  %s| %s     %s |%s"%(l,llbl,rlbl,r))
        print("       |             |     ")
        print("       ---------------     ")
        
        print("")


    def __str__(self):
        print(self.Storage)
        return ""

    def __repr__(self):
        print(self.Storage)
        return ""

    def __len__(self):
        return len(self.Storage)

    def shape(self):
        return self.Storage.shape
    
    ## Fill :
    def __getitem__(self,key):
        return self.Storage[key]

    def __setitem__(self,key,value):
        self.Storage[key] = value
         


    ## Math ::
    def __add__(self,other):
        if isinstance(other, self.__class__):
            return UniTensor(D_IN = self.D_IN,\
                             D_OUT=self.D_OUT,\
                             label=self.label,\
                             torch_tensor=self.Storage + other.Storage)
        else :
            return UniTensor(D_IN=self.D_IN,
                             D_OUT=self.D_OUT,\
                             label=self.label,\
                             torch_tensor=self.Storage + other)

    def __sub__(self,other):
        if isinstance(other, self.__class__):
            return UniTensor(D_IN=self.D_IN,\
                             D_OUT=self.D_OUT,\
                             label=self.label,\
                             torch_tensor=self.Storage - other.Storage)
        else :
            return UniTensor(D_IN=self.D_IN,\
                             D_OUT=self.D_OUT,\
                             label=self.label,\
                             torch_tensor=self.Storage - other)

    def __mul__(self,other):
        if isinstance(other, self.__class__):
            return UniTensor(D_IN=self.D_IN,\
                             D_OUT=self.D_OUT,\
                             label=self.label,\
                             torch_tensor=self.Storage * other.Storage)
        else :
            return UniTensor(D_IN=self.D_IN,\
                             D_OUT=self.D_OUT,\
                             label=self.label,\
                             torch_tensor=self.Storage * other)
    """
    def __truediv__(self,other):
        if isinstance(other, self.__class__):
            return UniTensor(self.D_IN,self.D_OUT,\
                             self.Label_IN,self.Label_OUT,\
                             self.Storage / other.Storage)
        else :
            return UniTensor(self.D_IN,self.D_OUT,\
                             self.Label_IN,self.Label_OUT,\
                             self.Storage / other)
    """


    ## Extended Assignment:
    def __iadd__(self,other):
        if isinstance(other, self.__class__):
            self.Storage += other.Storage
        else :
            self.Storage += other
        return self

    def __imul__(self,other):
        if isinstance(other, self.__class__):
            self.Storage *= other.Storage
        else :
            self.Storage *= other
    
        return self



    ## Miscellaneous
    def Rand(self):
        _Randomize(self)





###############################################################
#
# Action function 
#
##############################################################
def Contract(a,b):
    if isinstance(a,UniTensor) and isinstance(b,UniTensor):
        ## get same vector:
        same = list(set(a.label).intersection(b.label)) 
        if(len(same)):
            print("Dev")    


        else:
            ## direct product
            new_D_IN = a.D_IN + a.D_OUT
            new_D_OUT = b.D_IN + b.D_OUT
            new_label = a.label + b.label
            DALL = new_D_IN + new_D_OUT
            maper = np.concatenate([np.arange(len(a.D_IN)), len(a.label) + np.arange(len(b.D_IN)), len(a.D_IN) + np.arange(len(a.D_OUT)), len(a.label) + len(b.D_IN) + np.arange(len(b.D_OUT))] ).tolist()

            return UniTensor(D_IN=a.D_IN + b.D_IN,\
                            D_OUT=a.D_OUT + b.D_OUT,\
                            label=a.label[:len(a.D_IN)] + b.label[:len(b.D_IN)] + a.label[len(a.D_IN):] + b.label[len(b.D_IN):],\
                            torch_tensor=torch.ger(a.Storage.view(-1),b.Storage.view(-1)).reshape(DALL).permute(maper))
            
    else:
        raise Exception('Contract(a,b)', "[ERROR] a and b both have to be UniTensor")




## This is the private function 
def _Randomize(a):
    """
        This is the private function [action fucntion] for Randomize a UniTensor.
    """
    if isinstance(a,UniTensor):
        a.Storage = torch.rand(a.Storage.shape, dtype=a.Storage.dtype, device=a.Storage.device)
    else:
        raise Exception("_Randomize(UniTensor)","[ERROR] _Randomize can only accept UniTensor")

def _svd(a):
    """
        This is the private function [action function] for Svd a UniTensor. 
        The function performs the svd by merging all the in-bonds and out-bonds to singule bond repectivly.
        The return will be a two Unitary tensors with singular values represented as 1-rank UniTensor.
    """
    if isinstance(a,UniTensor):
        # u, s, v = torch.svd(a)
        print("developing.")
        exit(1)
        #return torch.svd(a)
    else:
        raise Exception("_Randomize(UniTensor)","[ERROR] _Randomize can only accept UniTensor")

# def _qr(a):





 
