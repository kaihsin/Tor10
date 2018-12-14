import torch 
import copy
import numpy as np

from .Bond import *


class UniTensor():

    def __init__(self, bonds, labels=None, device=torch.device("cpu"),dtype=torch.float64,torch_tensor=None,check=True):
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

        self.bonds = np.array(copy.deepcopy(bonds))
                    
        
        if labels is None:
            self.labels = np.arange(len(self.bonds))
        else:
            self.labels = np.array(copy.copy(labels))
        
        ## Checking:
        if check:
            if not len(self.labels) == (len(self.bonds)):
                raise Exception("UniTensor.__init__","labels size is not consistence with the rank")
            if not len(np.unique(self.labels)) == len(self.labels):
                raise Exception("UniTensor.__init__","labels contain duplicate element.")


            ## sort all BD_IN on first and BD_OUT on last:
            #lambda x: 1 if x.bondType is BD_OUT else 0
            maper = np.argsort([ (x.bondType is BD_OUT) for x in self.bonds])
            self.bonds = self.bonds[maper]
            self.labels = self.labels[maper]

        if torch_tensor is None:
            DALL = [self.bonds[i].dim for i in range(len(self.bonds))]
            self.Storage = torch.zeros(tuple(DALL), device=device, dtype = dtype)
            del DALL
        else:
            self.Storage = torch_tensor
    


    def SetLabel(self, newLabel, idx):
        if not type(newLabel) is int or not type(idx) is int:
            raise TypeError("UniTensor.SetLabel","newLabel and idx must be int.")
        
        if not idx < len(self.labels):
            raise ValueError("UniTensor.SetLabel","idx exceed the number of bonds.")
        
        if newLabel in self.labels:
            raise ValueError("UniTensor.SetLabel","newLabel [%d] already exists in the current UniTensor."%(newLabel))
        
        self.labels[idx] = newLabel
    
    def SetLabels(self,newlabels):
        if not len(newlabels) == len(self.labels):
            raise ValueError("UniTensor.SetLabels","the length of newlabels not match with the rank of UniTensor")
        
        if np.unique(newlabels) != len(newlabels):
            raise ValueError("UniTensor.SetLabels","the newlabels contain duplicated elementes.")

        self.labels = copy.copy(newlabels)


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
                                0  --| 1     3 |-- 3
                                     |         |
                                5  --| 2     6 |-- 11
                                     -----------

        """
        print("tensor Rank : %d"%(len(self.labels)))
        print("on device   : %s"%(self.Storage.device))        
        print("")        
        
        Nin = len([ 1 for i in range(len(self.labels)) if self.bonds[i].bondType is BD_IN])
        Nout = len(self.labels) - Nin    
    
        if Nin > Nout:
            vl = Nin
        else:
            vl = Nout

        print(vl)
        print("        ---------------     ")
        for i in range(vl):
            print("        |             |     ")
            if(i<Nin):
                l = "%3d __"%(self.labels[i])
                llbl = "%-3d"%(self.bonds[i].dim) 
            else:
                l = "      "
                llbl = "   "
            if(i<Nout):
                r = "__ %-3d"%(self.labels[Nin+i])
                rlbl = "%3d"%(self.bonds[Nin+i].dim)
            else:
                r = "      "
                rlbl = "   "
            print("  %s| %s     %s |%s"%(l,llbl,rlbl,r))
        print("        |             |     ")
        print("        ---------------     ")
        
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
            return UniTensor(bonds = self.bonds,\
                             labels= self.labels,\
                             torch_tensor=self.Storage + other.Storage,\
                             check=False)
        else :
            return UniTensor(bonds = self.bonds,\
                             labels= self.labels,\
                             torch_tensor=self.Storage + other,\
                             check=False)

    def __sub__(self,other):
        if isinstance(other, self.__class__):
            return UniTensor(bonds = self.bonds,\
                             labels= self.labels,\
                             torch_tensor=self.Storage - other.Storage,\
                             check=False)
        else :
            return UniTensor(bonds = self.bonds,\
                             labels= self.labels,\
                             torch_tensor=self.Storage - other,\
                             check=False)

    def __mul__(self,other):
        if isinstance(other, self.__class__):
            return UniTensor(bonds = self.bonds,\
                             labels= self.labels,\
                             torch_tensor=self.Storage * other.Storage,\
                             check=False)
        else :
            return UniTensor(bonds = self.bonds,\
                             labels= self.labels,\
                             torch_tensor=self.Storage * other,\
                             check=False)

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

    ## This is the same function that behaves as the memberfunction.
    def Svd(self):
        return Svd(self)

    def Matmul(self,b):
        return Matmul(self,b)

    
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

    def CombineBonds(self,labels_to_combine):
        _CombineBonds(self,labels_to_combine)

    def Contiguous(self):
        self.Storage = self.Storage.contiguous()

    def is_contiguous(self):
        return self.Storage.is_contiguous()        

###############################################################
#
# Action function 
#
##############################################################
def Contract(a,b):
    if isinstance(a,UniTensor) and isinstance(b,UniTensor):
        ## get same vector:
        same = list(set(a.labels).intersection(b.labels)) 
        if(len(same)):
            print("Dev")    


        else:
            ## direct product
            Nin_a = len([1 for i in range(len(a.labels)) if a.bonds[i].bondType is BD_IN])
            Nin_b = len([1 for i in range(len(b.labels)) if b.bonds[i].bondType is BD_IN])
            Nout_a = len(a.labels) - Nin_a
            Nout_b = len(b.labels) - Nin_b

            new_label = np.concatenate([a.labels, b.labels])
            DALL = [a.bonds[i].dim for i in range(len(a.bonds))] + [b.bonds[i].dim for i in range(len(b.bonds))]

            maper = np.concatenate([np.arange(Nin_a), len(a.labels) + np.arange(Nin_b), np.arange(Nout_a) + Nin_a, len(a.labels) + Nin_b + np.arange(Nout_b)])

            return UniTensor(bonds=np.concatenate([a.bonds[:Nin_a],b.bonds[:Nin_b],a.bonds[Nin_a:],b.bonds[Nin_b:]]),\
                            labels=np.concatenate([a.labels[:Nin_a], b.labels[:Nin_b], a.labels[Nin_a:], b.labels[Nin_b:]]),\
                            torch_tensor=torch.ger(a.Storage.view(-1),b.Storage.view(-1)).reshape(DALL).permute(maper.tolist()),\
                            check=False)
            
    else:
        raise Exception('Contract(a,b)', "[ERROR] a and b both have to be UniTensor")


def Chain_matmul(*args):
    isUT = all( isinstance(UT,UniTensor) for UT in args)    
    
    tmp_args = [args[i].Storage for i in range(len(args))] 

    ## Checking performance:
    """  
    for i in range(len(tmp_args)):
        if not tmp_args[i] is args[i].Storage:
           print("Fatal performance")
           exit(1) 
    """

    if isUT:
        return UniTensor(bonds =[args[0].bonds[0],args[-1].bonds[1]],\
                         labels=[args[0].labels[0],args[-1].labels[1]],\
                         torch_tensor=torch.chain_matmul(*tmp_args),\
                         check=False)

    else:
        raise TypeError("_Chain_matmul(*args)", "[ERROR] _Chain_matmul can only accept UniTensors for all elements in args")

def Matmul(a,b):
    if isinstance(a,UniTensor) and isinstance(b,UniTensor):

        ## no need to check if a,b are both rank 2. Rely on torch to do error handling! 
        return UniTensor(bonds =[a.bonds[0],b.bonds[1]],\
                         labels=[a.labels[0],b.labels[1]],\
                         torch_tensor=torch.matmul(a.Storage,b.Storage),\
                         check=False)


    else:
        raise TypeError("_Matmul(a,b)", "[ERROR] _Matmul can only accept UniTensors for both a & b")



def Svd(a):
    """
        This is the private function [action function] for Svd a UniTensor. 
        The function performs the svd by merging all the in-bonds and out-bonds to singule bond repectivly.
        The return will be a two Unitary tensors with singular values represented as 1-rank UniTensor.
    """
    if isinstance(a,UniTensor):
        if not len(a.labels) == 2:
            raise Exception("_svd","[ERROR] _svd can only accept UniTensor with rank 2")

        u, s, v = torch.svd(a.Storage,some=True)

        tmp = np.argwhere(a.labels<0)
        if len(tmp) == 0:
            tmp = 0
        else:
            tmp = np.min(tmp)

        u = UniTensor(bonds =[Bond(BD_IN,u.shape[0]),Bond(BD_OUT,u.shape[1])],\
                      labels=[a.labels[0],tmp-1],\
                      torch_tensor=u,\
                      check=False)
        v = UniTensor(bonds =[Bond(BD_IN,v.shape[1]),Bond(BD_OUT,v.shape[0])],\
                      labels=[tmp-2,a.labels[1]],\
                      torch_tensor=v.transpose(0,1),\
                      check=False)
        s = UniTensor(bonds  =[u.bonds[1],v.bonds[0]],\
                      labels =[u.labels[1],v.labels[0]],\
                      torch_tensor=torch.diag(s),\
                      check=False)   
        return u,s,v
    else:
        raise Exception("_svd(UniTensor)","[ERROR] _svd can only accept UniTensor")





## The functions that start with "_" are the private functions

def _CombineBonds(a,label):    
    if isinstance(a,UniTensor):
        if len(label) > len(a.labels):
            raise ValueError("_CombineBonds","[ERROR] the # of label_to_combine should be <= rank of UniTensor")
        # checking :
        same_lbls, x_ind, y_ind = np.intersect1d(a.labels,label,return_indices=True)
        #print(x_ind)
        #print(y_ind)
        #print(same_lbls)
        if not len(same_lbls) == len(label):
            raise Exception("_CombineBonds","[ERROR], label_to_combine doesn't exists in the UniTensor")
        
        idx_no_combine = np.setdiff1d(np.arange(len(a.labels)),x_ind)
        old_shape = np.array(a.Storage.shape)

        combined_dim = old_shape[x_ind]
        combined_dim = np.prod(combined_dim)
        no_combine_dims = old_shape[idx_no_combine]

        ## check if the combined bond will be in-bond or out-bond
        if a.bonds[x_ind[0]].bondType is BD_OUT:        
            maper = np.concatenate([idx_no_combine,x_ind])
            a.bonds = np.append(a.bonds[idx_no_combine],Bond(BD_OUT,combined_dim))
            a.labels = np.append(a.labels[idx_no_combine], a.labels[x_ind[0]])
            a.Storage = a.Storage.permute(maper.tolist()).reshape(np.append(no_combine_dims,combined_dim).tolist())
        else:
            maper = np.concatenate([x_ind,idx_no_combine])
            a.bonds = np.append(Bond(BD_IN,combined_dim),a.bonds[idx_no_combine])
            a.labels = np.append(a.labels[x_ind[0]],a.labels[idx_no_combine])
            a.Storage = a.Storage.permute(maper.tolist()).reshape(np.append(combined_dim,no_combine_dims).tolist())





    else :
        raise Exception("_CombineBonds(UniTensor,int_arr)","[ERROR] )CombineBonds can only accept UniTensor")

def _Randomize(a):
    """
        This is the private function [action fucntion] for Randomize a UniTensor.
    """
    if isinstance(a,UniTensor):
        a.Storage = torch.rand(a.Storage.shape, dtype=a.Storage.dtype, device=a.Storage.device)
    else:
        raise Exception("_Randomize(UniTensor)","[ERROR] _Randomize can only accept UniTensor")







 
