import torch 
import copy,os
import numpy as np
import pickle as pkl
from .Bond import *


## Developer Note:
# [KHW]
# Currently trying to add the Symm. 
# A temporary Abort is use to prevent the user to call the un-support operations on a Symmetry tensor. 
#
#  Find "Qnum_ipoint" keyword for the part that need to be modify accrodingly when considering the Qnums feature. 
#



class UniTensor():

    def __init__(self, bonds, labels=None, device=torch.device("cpu"),dtype=torch.float64,torch_tensor=None,check=True, is_diag=False, name=""):
        """
            @description: This is the initialization of the UniTensor.
            @param      : bonds [require]: The list of bonds. it should be an list or np.ndarray with len(list) is the # of bonds.  
                          label [option ]: The customize label. the number of elements should be the same as the total rank of the tensor, and contain on duplicated elements.
                          device[option ]: This should be a [torch.device]. When provided, the tensor will be put on the device ("cpu", "cuda", "cuda:x" with x is the GPU-id. See torch.device for further information.)
                          dtype [option ]: This should be a [torch.dtype ]. The default type is float with either float32 or float64 which follows the same internal rule of pytorch. For further information, see pytorch documentation. 
                          torch_tensor [private]: This is the internal arguments in current version. It should not be directly use, otherwise may cause inconsistence with Bonds and memory layout. 
                                                  ** Developer **
                                                  > The torch_tensor should have the same rank as len(label), and with each bond dimensions strictly the same as describe as in bond in self.bonds.
                          check [private]: This is the internal arguments. It should not be directly use. If False, all the checking across bonds/labels/Storage.shape will be ignore. 
                          is_diag [option]: This states the current UniTensor is a diagonal matrix or not. If True, the Storage will only store diagonal elements. 
                          name [option]: This states the name of current UniTensor.      
        """
        self.bonds = np.array(copy.deepcopy(bonds))
        self.name = name
        self.is_diag = is_diag        

        
        if labels is None:
            self.labels = np.arange(len(self.bonds))
        else:
            self.labels = np.array(copy.deepcopy(labels))
        
        ## Checking:
        if check:
            if not len(self.labels) == (len(self.bonds)):
                raise Exception("UniTensor.__init__","labels size is not consistence with the rank")
            if not len(np.unique(self.labels)) == len(self.labels):
                raise Exception("UniTensor.__init__","labels contain duplicate element.")
            if is_diag:
                if not len(self.labels) == 2:
                    raise TypeError("UniTensor.__init__","is_diag=True require Tensor rank==2")
                if not self.bonds[0].dim == self.bonds[1].dim:
                    raise TypeError("UniTensor.__init__","is_diag=True require Tensor to be square rank-2")
            if len(np.unique([ (bd.qnums is None) for bd in self.bonds])) != 1:
                raise TypeError("UniTensor.__init__","the bonds are not consistent. Cannot have mixing bonds of with and without symmetry (qnums).")

            ## sort all BD_IN on first and BD_OUT on last:
            #lambda x: 1 if x.bondType is BD_OUT else 0
            maper = np.argsort([ (x.bondType is BD_OUT) for x in self.bonds])
            self.bonds = self.bonds[maper]
            self.labels = self.labels[maper]
            
            


        if torch_tensor is None:
            if self.is_diag:
                self.Storage = torch.zeros(self.bonds[0].dim,device=device,dtype=dtype)                
            else:
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
        if isinstance(newlabels,list):
            newlabels = np.array(newlabels)

        if not len(newlabels) == len(self.labels):
            raise ValueError("UniTensor.SetLabels","the length of newlabels not match with the rank of UniTensor")
        
        if len(np.unique(newlabels)) != len(newlabels):
            raise ValueError("UniTensor.SetLabels","the newlabels contain duplicated elementes.")

        self.labels = copy.copy(newlabels)

    def SetElem(self, elem):
        """
        @Description: Given 1D array of elements, set the elements stored in tensor as the same as the given ones. Note that elem can only be python-list or numpy 
        
        """
        if not isinstance(elem,list) and not isinstance(elem,np.ndarray):
            raise TypeError("UniTensor.SetElem","[ERROR]  elem can only be python-list or numpy")
        
        if not len(elem) == self.Storage.numel():
            raise ValueError("UniTensor.SetElem","[ERROR] number of elem is not equal to the # of elem in the tensor.")
        

        ## Qnum_ipoint
        if self.bonds[0].qnums is not None:
            raise Exception("UniTensor.SetElem","[Abort] the TN that has symm is under developing.")
        
        my_type = self.Storage.dtype
        my_shape = self.Storage.shape
        my_device = self.Storage.device
        self.Storage = torch.from_numpy(np.array(elem)).type(my_type).reshape(my_shape).to(my_device)
        
    def Todense(self):
        if self.is_diag==True:
            self.Storage = torch.diag(self.Storage) 
            self.is_diag=False

    def to(self,device):
        if not isinstance(device,torch.device):
            raise TypeError("[ERROR] UniTensor.to()","only support device argument in this version as torch.device")
        self.Storage = self.Storage.to(device)         

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
        print("tensor Name : %s"%(self.name))
        print("tensor Rank : %d"%(len(self.labels)))
        print("on device   : %s"%(self.Storage.device))        
        print("is_diag     : %s"%("True" if self.is_diag else "False"))        
        
        Nin = len([ 1 for i in range(len(self.labels)) if self.bonds[i].bondType is BD_IN])
        Nout = len(self.labels) - Nin    
    
        if Nin > Nout:
            vl = Nin
        else:
            vl = Nout

        #print(vl)
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
        
        for i in range(len(self.bonds)):
            print("lbl:%d "%(self.labels[i]),end="")
            print(self.bonds[i])


        

    def __str__(self):
        print("Tensor name: %s"%( self.name))
        print("is_diag    : %s"%("True" if self.is_diag else "False"))
        print(self.Storage)
        return ""

    def __repr__(self):
        print("Tensor name: %s"%( self.name))
        print("is_diag    : %s"%("True" if self.is_diag else "False"))
        print(self.Storage)
        return ""

    def __len__(self):
        return len(self.Storage)

    def __eq__(self,rhs):
        """
            Note that this is only compare the shape of Storage. Not the content of torch tensor.
        """
        if isinstance(rhs,self.__class__):
            iss = (self.Storage.shape == rhs.Storage.shape) and (len(self.bonds) == len(rhs.bonds))
            if not iss:
                return False
            
            iss = iss and all(self.bonds[i]==rhs.bonds[i] for i in range(len(self.bonds))) and all(self.labels[i]==rhs.labels[i] for i in range(len(self.labels)))
            return iss
                
        else:
            raise ValueError("Bond.__eq__","[ERROR] invalid comparison between Bond object and other type class.")


    def shape(self):
        if self.is_diag:
            return torch.Size([self.bonds.dim[0],self.bonds.dim[0]])
        else:
            return self.Storage.shape
    
    ## Fill :
    def __getitem__(self,key):
        return self.Storage[key]

    def __setitem__(self,key,value):
        self.Storage[key] = value
         


    ## Math ::
    def __add__(self,other):
        if isinstance(other, self.__class__):
            if self.is_diag and other.is_diag:
                tmp = UniTensor(bonds = self.bonds,\
                                labels= self.labels,\
                                torch_tensor=self.Storage + other.Storage,\
                                check=False,\
                                is_diag=True)

            elif self.is_diag==False and other.is_diag==False:
                tmp = UniTensor(bonds = self.bonds,\
                                labels= self.labels,\
                                torch_tensor=self.Storage + other.Storage,\
                                check=False)
            else:
                if self.is_diag:
                    tmp = UniTensor(bonds = self.bonds,\
                                    labels= self.labels,\
                                    torch_tensor=torch.diag(self.Storage) + other.Storage,\
                                    check=False)
                else:
                    tmp = UniTensor(bonds = self.bonds,\
                                    labels= self.labels,\
                                    torch_tensor=self.Storage + torch.diag(other.Storage),\
                                    check=False)
            return tmp
        else:
            return UniTensor(bonds = self.bonds,\
                             labels= self.labels,\
                             torch_tensor=self.Storage + other,\
                             check=False,
                             is_diag=self.is_diag)

    def __sub__(self,other):
        if isinstance(other, self.__class__):
            if self.is_diag and other.is_diag:
                tmp = UniTensor(bonds = self.bonds,\
                                labels= self.labels,\
                                torch_tensor=self.Storage - other.Storage,\
                                check=False,\
                                is_diag=True)

            elif self.is_diag==False and other.is_diag==False:
                tmp = UniTensor(bonds = self.bonds,\
                                 labels= self.labels,\
                                 torch_tensor=self.Storage - other.Storage,\
                                 check=False)
            else:
                if self.is_diag:
                    tmp = UniTensor(bonds = self.bonds,\
                                    labels= self.labels,\
                                    torch_tensor=torch.diag(self.Storage) - other.Storage,\
                                    check=False)
                else:
                    tmp = UniTensor(bonds = self.bonds,\
                                    labels= self.labels,\
                                    torch_tensor=self.Storage - torch.diag(other.Storage),\
                                    check=False)
            return tmp
        else :
            return UniTensor(bonds = self.bonds,\
                             labels= self.labels,\
                             torch_tensor=self.Storage - other,\
                             check=False,
                             is_diag=self.is_diag)

    def __mul__(self,other):
        if isinstance(other, self.__class__):
            if self.is_diag and other.is_diag:
                tmp = UniTensor(bonds = self.bonds,\
                                labels= self.labels,\
                                torch_tensor=self.Storage * other.Storage,\
                                check=False,\
                                is_diag=True)

            elif self.is_diag==False and other.is_diag==False:
                tmp = UniTensor(bonds = self.bonds,\
                                 labels= self.labels,\
                                 torch_tensor=self.Storage * other.Storage,\
                                 check=False)
            else:
                if self.is_diag:
                    tmp = UniTensor(bonds = self.bonds,\
                                    labels= self.labels,\
                                    torch_tensor=torch.diag(self.Storage) * other.Storage,\
                                    check=False)
                else:
                    tmp = UniTensor(bonds = self.bonds,\
                                    labels= self.labels,\
                                    torch_tensor=self.Storage * torch.diag(other.Storage),\
                                    check=False)
        else:
            tmp = UniTensor(bonds = self.bonds,\
                            labels= self.labels,\
                            torch_tensor=self.Storage * other,\
                            check=False,\
                            is_diag=self.is_diag)
        return tmp

    def __pow__(self,other):
        return UniTensor(bonds=self.bonds,\
                         labels=self.labels,\
                         torch_tensor=self.Storage**other,\
                         check=False,\
                         is_diag=self.is_diag)

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

    def Svd_truncate(self):
        
        return Svd_truncate(self)

    def Norm(self):
        return Norm(self)

    def Det(self):
        return Det(self)

    def Matmul(self,b):
        
        return Matmul(self,b)

    
    ## Extended Assignment:
    def __iadd__(self,other):
        if isinstance(other, self.__class__):
            if self.is_diag == other.is_diag:            
                self.Storage += other.Storage
            else:
                if self.is_diag:
                    self.Storage = torch.diag(self.Storage) + other.Storage
                    self.is_diag=False
                else:
                    self.Storage += torch.diag(other.Storage)

        else :
            self.Storage += other
        return self

    def __isub__(self,other):
        if isinstance(other, self.__class__):
            if self.is_diag == other.is_diag:            
                self.Storage -= other.Storage
            else:
                if self.is_diag:
                    self.Storage = torch.diag(self.Storage) + other.Storage
                    self.is_diag=False
                else:
                    self.Storage -= torch.diag(other.Storage)

        else :
            self.Storage -= other
        return self


    def __imul__(self,other):
        if isinstance(other, self.__class__):
            if self.is_diag == other.is_diag:            
                self.Storage *= other.Storage
            else:
                if self.is_diag:
                    self.Storage = torch.diag(self.Storage) * other.Storage
                    self.is_diag=False
                else:
                    self.Storage *= torch.diag(other.Storage)
        else :
            self.Storage *= other
    
        return self


    ## Miscellaneous
    def Rand(self):
        ## Qnum_ipoint
        if self.bonds[0].qnums is not None:
            raise Exception("[Abort] UniTensor.Rand for symm TN is under developing")

        _Randomize(self)

    def CombineBonds(self,labels_to_combine):
        _CombineBonds(self,labels_to_combine)

    def Contiguous(self):
        self.Storage = self.Storage.contiguous()

    def is_contiguous(self):
        return self.Storage.is_contiguous()        


    def Permute(self,maper,N_inbond,by_label=False):
        if self.is_diag:
            raise Exception("UniTensor.Permute","[ERROR] UniTensor.is_diag=True cannot be permuted.\n"+
                                                "[Suggest] Call UniTensor.Todense()")
        if not isinstance(maper,list):
            raise TypeError("UniTensor.Permute","[ERROR] maper should be an python list.")            
 
       
        if by_label:
            ## check all label
            if not all(lbl in self.labels for lbl in maper):
                raise Exception("UniTensor.Permute","[ERROR] by_label=True but maper contain invalid labels not appear in the UniTensor label")

            DD = dict(zip(self.labels,np.arange(len(self.labels))))
            new_maper=[ DD[x] for x in maper]
            self.Storage = self.Storage.permute(new_maper)
            self.labels = np.array(maper)
            self.bonds = self.bonds[new_maper]

        else:
            ## We don't need this. pytorch will handle the dimesion mismatch error.
            #if not len(maper) == len(self.labels):
            #    raise ValueError("UniTensor.Permute", "[ERROR] len of maper should be the same as the rank of the UniTensor.")

            self.Storage = self.Storage.permute(maper)
            self.labels = self.labels[maper]
            self.bonds = self.bonds[maper]

        for i in range(len(self.bonds)):
            if i < N_inbond:
                self.bonds[i].change(BD_IN)
            else:
                self.bonds[i].change(BD_OUT)


    def Reshape(self,dimer,N_inbond,new_labels=None):
        if self.is_diag:
            raise Exception("UniTensor.Reshape","[ERROR] UniTensor.is_diag=True cannot be Reshape.\n"+
                                                "[Suggest] Call UniTensor.Todense()")
        if not isinstance(dimer,list):
            raise TypeError("UniTensor.Reshape","[ERROR] maper should be an python list.")            

        ## Qnum_ipoint
        if self.bonds[0].qnums is not None:
            raise Exception("UniTensor.Reshape","[Abort] UniTensor with symm cannot be Reshape.\n")


        ## This is not contiguous
        self.Storage = self.Storage.view(dimer)
        if new_labels is None:
            self.labels = np.arange(len(dimer))
        else:
            self.labels  = np.array(new_labels)

        f = lambda i,Nid,dim : Bond(BD_IN,dim) if i<Nid else Bond(BD_OUT,dim)
        self.bonds  = np.array([f(i,N_inbond,dimer[i]) for i in range(len(dimer))])


    ## Symmetric Tensor function
    def GetBlock(self,qnum=None):
        if qnum is not None:
            ## check if symm:
            if self.bonds[0].qnums is None:
                raise TypeError("UniTensor.GetBlock","[ERROR] Trying to get a block on a non-symm tensor.")

            if self.is_diag:
                raise TypeError("UniTensor.GetBlock","[ERROR] Cannot get block on a diagonal tensor (is_diag=True)")

            picker = [np.argwhere(self.bonds[i].qnums==qnum).flatten() for i in range(len(self.bonds))]
            #print(picker)
            
            return UniTensor(bonds=[Bond(self.bonds[i].bondType,dim=len(picker[i])) for i in range(len(self.bonds))],\
                             labels=self.labels,\
                             torch_tensor=self.Storage[np.ix_(*picker)],\
                             check=False)
        else:
            print("[Warning] GetBlock a non-symmetry TN will return self.")
            return self



###############################################################
#
# Action function 
#
##############################################################
## I/O
def Save(a,filename):
    if not isinstance(filename,str):
        raise TypeError("Save","[ERROR] Invalid filename.")
    if not isinstance(a,UniTensor):
        raise TypeError("Save","[ERROR] input must be the UniTensor")
    f = open(filename,"wb")
    pkl.dump(a,f)
    f.close()

def Load(filename):
    if not isinstance(filename,str):
        raise TypeError("UniTensor.Save","[ERROR] Invalid filename.")
    if not os.path.exists(filename):
        raise Exception("UniTensor.Load","[ERROR] file not exists")

    f = open(filename,'rb')
    tmp = pkl.load(f)
    f.close()
    if not isinstance(tmp,UniTensor):
        raise TypeError("Load","[ERROR] loaded object is not the UniTensor")
    
    return tmp


def Contract(a,b):
    if isinstance(a,UniTensor) and isinstance(b,UniTensor):
        ## Qnum_ipoint
        if a.bonds[0].qnums is not None or b.bonds[0].qnums is not None:
           raise Exception("Contract(a,b)","[Abort] contract Symm TN is under developing.")


        ## get same vector:
        same, a_ind, b_ind = np.intersect1d(a.labels,b.labels,return_indices=True)

        if(len(same)):
            aind_no_combine = np.setdiff1d(np.arange(len(a.labels)),a_ind)
            bind_no_combine = np.setdiff1d(np.arange(len(b.labels)),b_ind)
            
            maper_a = np.concatenate([aind_no_combine,a_ind])
            maper_b = np.concatenate([b_ind,bind_no_combine])

            old_shape = np.array(a.Storage.shape) if a.is_diag==False else np.array([a.Storage.shape[0],a.Storage.shape[0]])
            combined_dim = np.prod(old_shape[a_ind])

            if a.is_diag :
                tmpa = torch.diag(a.Storage).to(a.Storage.device)
            else:   
                tmpa = a.Storage
            
            if b.is_diag :
                tmpb = torch.diag(b.Storage).to(b.Storage.device)
            else:   
                tmpb = b.Storage


            tmp = torch.matmul(tmpa.permute(maper_a.tolist()).reshape(-1,combined_dim),\
                               tmpb.permute(maper_b.tolist()).reshape(combined_dim,-1))
            new_shape = [ bd.dim for bd in a.bonds[aind_no_combine]] + [ bd.dim for bd in b.bonds[bind_no_combine]]
            return UniTensor(bonds =np.concatenate([a.bonds[aind_no_combine],b.bonds[bind_no_combine]]),\
                             labels=np.concatenate([a.labels[aind_no_combine],b.labels[bind_no_combine]]),\
                             torch_tensor=tmp.view(new_shape),\
                             check=False)

        else:
            ## direct product
            Nin_a = len([1 for i in range(len(a.labels)) if a.bonds[i].bondType is BD_IN])
            Nin_b = len([1 for i in range(len(b.labels)) if b.bonds[i].bondType is BD_IN])
            Nout_a = len(a.labels) - Nin_a
            Nout_b = len(b.labels) - Nin_b

            new_label = np.concatenate([a.labels, b.labels])
            DALL = [a.bonds[i].dim for i in range(len(a.bonds))] + [b.bonds[i].dim for i in range(len(b.bonds))]

            maper = np.concatenate([np.arange(Nin_a), len(a.labels) + np.arange(Nin_b), np.arange(Nout_a) + Nin_a, len(a.labels) + Nin_b + np.arange(Nout_b)])

            if a.is_diag :
                tmpa = torch.diag(a.Storage)
            else:   
                tmpa = a.Storage
            
            if b.is_diag :
                tmpb = torch.diag(b.Storage)
            else:   
                tmpb = b.Storage


            return UniTensor(bonds=np.concatenate([a.bonds[:Nin_a],b.bonds[:Nin_b],a.bonds[Nin_a:],b.bonds[Nin_b:]]),\
                            labels=np.concatenate([a.labels[:Nin_a], b.labels[:Nin_b], a.labels[Nin_a:], b.labels[Nin_b:]]),\
                            torch_tensor=torch.ger(tmpa.view(-1),tmpb.view(-1)).reshape(DALL).permute(maper.tolist()),\
                            check=False)
            
    else:
        raise Exception('Contract(a,b)', "[ERROR] a and b both have to be UniTensor")


def Chain_matmul(*args):
    """
        @description: This function performs matrix multiplication on all the UniTensors. Note that all the UniTensors should be rank-2 with 1-inbond 1-outbond

        @params     : The UniTensors that will be matrix-multiply

        @return     : UniTensor,rank-2, 1 inbond 1 outbond. The label of inbond = the label of inbond of first UniTensor. The label of outbond = the label of outbond of the last UniTensor.
        @exampe     : 
                        f = Chain_matmul(a,b,c,d,e)
                        Mathmatically equivalent as : f = a \cdot b \cdot c \cdot d \cdot e

    """
    f = lambda x,idiag: torch.diag(x) if idiag else x 
    isUT = all( isinstance(UT,UniTensor) for UT in args)    
    
    
    tmp_args = [f(args[i].Storage,args[i].is_diag) for i in range(len(args))] 

    ## Checking performance:
    """  
    for i in range(len(tmp_args)):
        if not tmp_args[i] is args[i].Storage:
           print("Fatal performance")
           exit(1) 
    """

    if isUT:
        ## Qnum_ipoint
        if not all( (UT.bonds[0].qnums is None) for UT in args):
            raise Exception("Chain_matmul(*args)","[Abort] Chain multiplication for symm tensor(s) are under developing.")


        return UniTensor(bonds =[args[0].bonds[0],args[-1].bonds[1]],\
                         labels=[args[0].labels[0],args[-1].labels[1]],\
                         torch_tensor=torch.chain_matmul(*tmp_args),\
                         check=False)

    else:
        raise TypeError("_Chain_matmul(*args)", "[ERROR] _Chain_matmul can only accept UniTensors for all elements in args")

def Matmul(a,b):
    
    if isinstance(a,UniTensor) and isinstance(b,UniTensor):

        ## [Note] no need to check if a,b are both rank 2. Rely on torch to do error handling! 

        ## Qnum_ipoint
        if a.bonds[0].qnums is not None or b.bonds[0].qnums is not None:
            raise Exception("Matmul(a,b)","[Abort] Matmul for sym TN is under developing.")

        if a.is_diag == b.is_diag:
            tmp = UniTensor(bonds =[a.bonds[0],b.bonds[1]],\
                            labels=[a.labels[0],b.labels[1]],\
                            torch_tensor=torch.matmul(a.Storage,b.Storage),\
                            check=False,\
                            is_diag=a.is_diag)
        else:
            if a.is_diag:
                tmp = UniTensor(bonds =[a.bonds[0],b.bonds[1]],\
                                labels=[a.labels[0],b.labels[1]],\
                                torch_tensor=torch.matmul(torch.diag(a.Storage),b.Storage),\
                                check=False)
            if b.is_diag:
                tmp = UniTensor(bonds =[a.bonds[0],b.bonds[1]],\
                                labels=[a.labels[0],b.labels[1]],\
                                torch_tensor=torch.matmul(a.Storage,torch.diag(b.Storage)),\
                                check=False)

        return tmp

    else:
        raise TypeError("_Matmul(a,b)", "[ERROR] _Matmul can only accept UniTensors for both a & b")



def Svd(a):
    """
        @description : The function performs the svd to input UniTensor [a]. The UniTensor should be rank-2 with 1-inbond 1-outbond. each inbond and outbond's dim should be >=2. 
                       Mathmatically, a = u \cdot s \cdot vt
        @params      :  a : UniTensor, rank-2, 1 inbond 1 outbond.
        @return      :  u , s , vt 
                        u : UniTensor, 2-rank, 1 inbond 1 outbond, the unitary matrix
                        s : UniTensor, 2-rank, 1 inbond 1 outbond, the diagonal, singular matrix 
                        vt: UniTensor, 2-rank, 1 inbond 1 outbond, the transposed right unitary matrix
    """
    if isinstance(a,UniTensor):

        ## Qnum_ipoint
        if a.bonds[0].qnums is not None:
            raise Exception("svd(a)","[Abort] svd curretly don't support symm tensor.")


        if a.is_diag:
            raise Exception("svd(a)","[Abort] svd currently don't support diagonal tensor.")

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
                      torch_tensor=s,\
                      check=False,\
                      is_diag=True)   
        return u,s,v
    else:
        raise Exception("Svd(UniTensor)","[ERROR] Svd can only accept UniTensor")


def ExpH(a):
    """
        @description : This function performs the exp^{H} where H is the hermitian matrix. Intricate compute svd first and exp the singular matrix.
        @params      : a : UniTensor, rank-2
        @return      : UniTensor, 2-rank, same bonds and labels at the original H
    """

    if isinstance(a,UniTensor):
        ## Qnum_ipoint
        if a.bonds[0].qnums is not None:
            raise Exception("ExpH(a)","[Abort] curretly don't support symm tensor.")

        if a.is_diag:
            u = torch.exp(a.Storage)
            return UniTensor(bonds=a.bonds,\
                             labels=a.labels,\
                             torch_tensor=u,\
                             is_diag=True,\
                             check=False)
        else:
            ## version-1, only real, not sure if it can extend to complex
            s , u = torch.symeig(a.Storage,eigenvectors=True)
            s     = torch.exp(s)

            # torch.matmul(u*s,u.transpose(0,1),out=u)
            u = torch.matmul(u*s,u.transpose(0,1))
            del s

            return UniTensor(bonds=a.bonds,\
                            labels=a.labels,\
                            torch_tensor=u,\
                            check=False)
                
    else:
        raise Exception("ExpH(UniTensor)","[ERROR] ExpH can only accept UniTensor")





## The functions that start with "_" are the private functions

def _CombineBonds(a,label):    
    """
        @description : <Private function> This function combines the bonds in input UniTensor [a] by the specified labels [label]. The bondType of the combined bonds will always follows the same bondType of bond in [a] with label of the first element in [label] 
        @param       : 
                        a    : UniTensor
                        label: the labels that is being combined.

        @return      : N/A

    """
    if isinstance(a,UniTensor):
        if a.is_diag:
            raise TypeError("_CombineBonds","[ERROR] CombineBonds doesn't support diagonal matrix.")
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
            for i in range(len(x_ind)-1):
                a.bonds[x_ind[0]].combine(a.bonds[x_ind[1+i]])
            a.bonds = np.append(a.bonds[idx_no_combine],a.bonds[x_ind[0]])
            a.labels = np.append(a.labels[idx_no_combine], a.labels[x_ind[0]])
            a.Storage = a.Storage.permute(maper.tolist()).contiguous().view(np.append(no_combine_dims,combined_dim).tolist())
        else:
            maper = np.concatenate([x_ind,idx_no_combine])
            for i in range(len(x_ind)-1):
                a.bonds[x_ind[0]].combine(a.bonds[x_ind[1+i]])
            a.bonds = np.append(a.bonds[x_ind[0]],a.bonds[idx_no_combine])
            a.labels = np.append(a.labels[x_ind[0]],a.labels[idx_no_combine])
            a.Storage = a.Storage.permute(maper.tolist()).contiguous().view(np.append(combined_dim,no_combine_dims).tolist())





    else :
        raise Exception("_CombineBonds(UniTensor,int_arr)","[ERROR] )CombineBonds can only accept UniTensor")

def _Randomize(a):
    """
        @description: <private function> This function randomize a UniTensor.
        @params     : 
                      a : UniTensor
        @return     : N/A 
                    
    """

    if isinstance(a,UniTensor):
    
        a.Storage = torch.rand(a.Storage.shape, dtype=a.Storage.dtype, device=a.Storage.device)
    
        
    else:
        raise Exception("_Randomize(UniTensor)","[ERROR] _Randomize can only accept UniTensor")


def Qr(a):
    """
        @description : The function performs the qr to input UniTensor [a]. The UniTensor should be rank-2 with 1-inbond 1-outbond. each inbond and outbond's dim should be >=2. 
                       Mathmatically, a = q \cdot r
        @params      :  a : UniTensor, rank-2, 1 inbond 1 outbond.
        @return      :  q , r  
                        q : UniTensor, 2-rank, 1 inbond 1 outbond, the unitary matrix
                        r : UniTensor, 2-rank, 1 inbond 1 outbond, the upper triangular matrix 
    """
    if isinstance(a,UniTensor):

        ## Qnum_ipoint
        if a.bonds[0].qnums is not None:
            raise Exception("Qr(a)","[Abort] curretly don't support symm tensor.")

        if a.is_diag:
            raise Exception("Qr(UniTensor)","[Aboart] Currently not support diagonal tensors.")
        
        q, r = torch.qr(a.Storage)

        tmp = np.argwhere(a.labels<0)
        if len(tmp) == 0:
            tmp = 0
        else:
            tmp = np.min(tmp)

        q = UniTensor(bonds =[Bond(BD_IN,q.shape[0]),Bond(BD_OUT,q.shape[1])],\
                      labels=[a.labels[0],tmp-1],\
                      torch_tensor=q,\
                      check=False)
        r = UniTensor(bonds =[Bond(BD_IN,r.shape[0]),Bond(BD_OUT,r.shape[1])],\
                      labels=[q.labels[1],a.labels[1]],\
                      torch_tensor=r,\
                      check=False)
        return q,r
    else:
        raise Exception("Qr(UniTensor)","[ERROR] Qr can only accept UniTensor")


def Qdr(a):
    """
        @description : The function performs the qr to input UniTensor [a]. The UniTensor should be rank-2 with 1-inbond 1-outbond. each inbond and outbond's dim should be >=2. 
                       Mathmatically, a = q \cdot r
        @params      :  a : UniTensor, rank-2, 1 inbond 1 outbond.
        @return      :  q , r  
                        q : UniTensor, 2-rank, 1 inbond 1 outbond, the unitary matrix
                        r : UniTensor, 2-rank, 1 inbond 1 outbond, the upper triangular matrix 
    """
    if isinstance(a,UniTensor):

        ## Qnum_ipoint
        if a.bonds[0].qnums is not None:
            raise Exception("Qdr(a)","[Abort] curretly don't support symm tensor.")

        if a.is_diag:
            raise Exception("Qr(UniTensor)","[Aboart] Currently not support diagonal tensors.")

        q, r = torch.qr(a.Storage)
        d = r.diag()
        r = (r.t()/d).t()

        tmp = np.argwhere(a.labels<0)
        if len(tmp) == 0:
            tmp = 0
        else:
            tmp = np.min(tmp)

        q = UniTensor(bonds =[Bond(BD_IN,q.shape[0]),Bond(BD_OUT,q.shape[1])],\
                      labels=[a.labels[0],tmp-1],\
                      torch_tensor=q,\
                      check=False)
        d = UniTensor(bonds =[Bond(BD_IN,d.shape[0]),Bond(BD_OUT,d.shape[0])],\
                      labels=[tmp-1,tmp-2],\
                      torch_tensor=d,\
                      is_diag=True,
                      check=False)
        r = UniTensor(bonds =[Bond(BD_IN,r.shape[0]),Bond(BD_OUT,r.shape[1])],\
                      labels=[d.labels[1],a.labels[1]],\
                      torch_tensor=r,\
                      check=False)
        return q,d,r
    else:
        raise Exception("Qdr(UniTensor)","[ERROR] Qdr can only accept UniTensor")


def Svd_truncate(a, truncate=None):
    """
        @description : The function performs the svd to input UniTensor [a]. The UniTensor should be rank-2 with 1-inbond 1-outbond. each inbond and outbond's dim should be >=2. 
                       Mathmatically, a = u \cdot s \cdot vt
        @params      :  a : UniTensor, rank-2, 1 inbond 1 outbond.
        @return      :  u , s , vt 
                        u : UniTensor, 2-rank, 1 inbond 1 outbond, the unitary matrix
                        s : UniTensor, 2-rank, 1 inbond 1 outbond, the diagonal, singular matrix 
                        vt: UniTensor, 2-rank, 1 inbond 1 outbond, the transposed right unitary matrix
    """
    if isinstance(a,UniTensor):

        ## Qnum_ipoint
        if a.bonds[0].qnums is not None:
            raise Exception("Qdr(a)","[Abort] curretly don't support symm tensor.")

        if a.is_diag:
            raise Exception("svd(a)","[Abort] svd currently don't support diagonal tensor.")

        u, s, v = torch.svd(a.Storage,some=True)

        tmp = np.argwhere(a.labels<0)
        if len(tmp) == 0:
            tmp = 0
        else:
            tmp = np.min(tmp)

        if truncate is not None:
            if truncate < 0 or truncate > len(s):
                raise ValueError("Svd_truncate", "[ERROR] the truncate dimension is invalid")
            u = u[:, :truncate]
            s = s[:truncate]
            v = v[:, :truncate]

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
                      torch_tensor=s,\
                      check=False,\
                      is_diag=True)   
        return u,s,v
    else:
        raise Exception("Svd(UniTensor)","[ERROR] Svd can only accept UniTensor")


def Inverse(a):
    """
        @description: This function returns the inverse of a rank-2 tensor.
        @params     : 
                      a : UniTensor
        @return     : Unitensor
                    
    """
    if isinstance(a,UniTensor):
        
        if a.is_diag:
            a_inv = UniTensor(bonds = a.bonds,
                          labels=a.labels,
                          torch_tensor=a.Storage**-1,
                          is_diag=True,
                          check=False)
        else:
            a_inv = UniTensor(bonds = a.bonds,
                              labels=a.labels,
                              torch_tensor=torch.inverse(a.Storage),
                              check=False)
        return a_inv
    else:
        raise Exception("Inverse(UniTensor)","[ERROR] Inverse can only accept UniTensor")


def Det(a):
    """
        @description: This function returns the determinant a rank-2 tensor.
        @params     : 
                      a : rank-2 UniTensor
        @return     : a 0-dimension tensor contains the determinant of input
                    
    """
    if isinstance(a,UniTensor):

        if a.is_diag:
            return torch.prod(a.Storage)
        else:
            return torch.det(a.Storage)
    else:
        raise Exception("Det(UniTensor)","[ERROR] Det can only accept UniTensor")

def Norm(a):
    """
        @description: This function returns the frobinieus 2-norm of a tensor.
        @params     : 
                      a : UniTensor
        @return     : a 0-dimension tensor contains the 2-norm of input
                    
    """

    if isinstance(a,UniTensor):
        return torch.norm(a.Storage)
    else:
        raise Exception("Norm(UniTensor)","[ERROR] Norm can only accept UniTensor")
