import torch 
import copy,os
import numpy as np
import pickle as pkl
from .Bond import *
from .linalg import *

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
        This is the constructor of the UniTensor.

        Public Args:

            bonds: 
                The list of bonds. it should be an list or np.ndarray with len(list) is the # of bonds.  

            labels: 
                The label of each bond. 
                1. the number of elements should be the same as the total rank of the tensor, contain no duplicated elements.
                2. all the label should be integer. if the label is specify as floating point, it will be rounded as integer. 

            device: 
                This should be a [torch.device]. When provided, the tensor will be put on the device ("cpu", "cuda", "cuda:x" with x is the GPU-id. See torch.device for further information.)
            
            dtype : 
                This should be a [ torch.dtype ]. 
                *The default type is float with either float32 or float64 which follows the same internal rule of pytorch. For further information, see pytorch documentation. 
            
            is_diag: 
                This states if the current UniTensor is a diagonal matrix or not. If True, the Storage will only store diagonal elements.
                Note that if is_diag=True, then the UniTensor is strictly required to be a square 2-rank tensor.  
            
            name: 
                This states the name of current UniTensor.      

        Private Args:

            \color{red}{[Warning] Private Args should not be call directly}

            torch_tensor : 
                This is the internal arguments in current version. It should not be directly use, otherwise may cause inconsistence with Bonds and memory layout. 
                    *For Developer:
                        > The torch_tensor should have the same rank as len(label), and with each bond dimensions strictly the same as describe as in bond in self.bonds.

            check : 
                This is the internal arguments. It should not be directly use. If False, all the checking across bonds/labels/Storage.shape will be ignore. 
        

        Example for how to create a UniTensor:
        
            * create a 2-rank UniTensor (matrix) with shape (3,4): 
            >>> a = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,4)])

            * create a 3-rank UniTensor with shape (3,4,5) and set labels [-3,4,1] for each bond:
            >>> c = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,4),Tor10.Bond(Tor10.BD_OUT,5)],labels=[-3,4,1])

            * create a 2-rank UniTensor with shape (3,4) on GPU-0:
            >>> d = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,4)],device=torch.device("cuda:0"))

            * create a diagonal 6x6 2-rank tensor(matrix):
            >>> e = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,6),Tor10.Bond(Tor10.BD_OUT,6)],is_diag=True)
            
            Note that when is_diag is set to True, the UniTensor should be a square matrix.

            * crate a 3-rank UniTensor with single precision:
            >>> f = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,4),Tor10.Bond(Tor10.BD_OUT,5)],labels=[-3,4,1],dtype=torch.float32)
            


        """
        self.bonds = np.array(copy.deepcopy(bonds))
        self.name = name
        self.is_diag = is_diag        

        
        if labels is None:
            self.labels = np.arange(len(self.bonds))
        else:
            self.labels = np.array(copy.deepcopy(labels),dtype=np.int)
        
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

            if self.bonds[0].qnums is not None:
                if len(np.unique([ bd.nsym for bd in self.bonds])) != 1:
                    raise TypeError("UniTensor.__init__","the number of symmetry type for symmetry bonds doesn't match.")

            ## sort all BD_IN on first and BD_OUT on last:
            #lambda x: 1 if x.bondType is BD_OUT else 0
            #maper = np.argsort([ (x.bondType is BD_OUT) for x in self.bonds])
            #self.bonds = self.bonds[maper]
            #self.labels = self.labels[maper]


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
        """
        Set a new label for the bond local at specify index.

        Args:

            newLabel: The new label, it should be an integer.

            idx     : The index of the bond. when specified, the label of the bond at this index will be changed.

        Example:

            >>> a = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,4)],labels=[5,6])
            >>> a.labels
            [5 6]

            Set "-1" to replace the original label "6" at index 1
            >>> a.SetLabel(-1,1)
            >>> a.labels
            [5 -1]
 
        """
        if not type(newLabel) is int or not type(idx) is int:
            raise TypeError("UniTensor.SetLabel","newLabel and idx must be int.")
        
        if not idx < len(self.labels):
            raise ValueError("UniTensor.SetLabel","idx exceed the number of bonds.")
        
        if newLabel in self.labels:
            raise ValueError("UniTensor.SetLabel","newLabel [%d] already exists in the current UniTensor."%(newLabel))
        
        self.labels[idx] = newLabel
    
    def SetLabels(self,newlabels):
        """
        Set new labels for all the bonds.

        Args:

            newLabels: The list of new label, it should be an python list or numpy array with size equal to the number of bonds of the UniTensor.
                       
        Example:

            >>> a = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,4)],labels=[5,6])
            >>> a.labels
            [5 6]

            Set new_label=[-1,-2] to replace the original label [5,6].
            >>> new_label=[-1,-2]
            >>> a.SetLabels(new_label)
            >>> a.labels
            [-1 -2]
 
        """
        if isinstance(newlabels,list):
            newlabels = np.array(newlabels)

        if not len(newlabels) == len(self.labels):
            raise ValueError("UniTensor.SetLabels","the length of newlabels not match with the rank of UniTensor")
        
        if len(np.unique(newlabels)) != len(newlabels):
            raise ValueError("UniTensor.SetLabels","the newlabels contain duplicated elementes.")

        self.labels = copy.copy(newlabels)

    def SetElem(self, elem):
        """
        Given 1D array of elements, set the elements stored in tensor as the same as the given ones. Note that elem can only be python-list or numpy 
        
        Args:
            
            elem: 
                The elements to be replace the content of the current UniTensor. It should be a 1D array.
                **Note** if the UniTensor is a symmetric tensor, one should use UniTensor.PutBlock to set the elements.
 
        Example:
        ::
            Sz = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,2),Tt.Bond(Tt.BD_OUT,2)],
                              dtype=tor.float64,
                              device=tor.device("cpu"))
            Sz.SetElem([1, 0,
                        0,-1 ])

            >>> print(Sz)
            Tensor name: 
            is_diag    : False
            tensor([[ 1.,  0.],
                    [ 0., -1.]], dtype=torch.float64)
        
        """
        if not isinstance(elem,list) and not isinstance(elem,np.ndarray):
            raise TypeError("UniTensor.SetElem","[ERROR]  elem can only be python-list or numpy")
        
        if not len(elem) == self.Storage.numel():
            raise ValueError("UniTensor.SetElem","[ERROR] number of elem is not equal to the # of elem in the tensor.")
        
        ## Qnum_ipoint [OK]
        if self.bonds[0].qnums is not None:
            raise Exception("UniTensor.SetElem","[ERROR] the TN that has symm should use PutBlock.")
        
        my_type = self.Storage.dtype
        my_shape = self.Storage.shape
        my_device = self.Storage.device
        self.Storage = torch.from_numpy(np.array(elem)).type(my_type).reshape(my_shape).to(my_device)
        
    def Todense(self):
        """
        Set the UniTensor to dense matrix. Currently only the diagonal matrix is stored as sparsed form. So it only has effect on UniTensor where is_diag = True

        """
        if self.is_diag==True:
            self.Storage = torch.diag(self.Storage) 
            self.is_diag=False

    def to(self,device):
        """
        Set the UniTensor to device

        Args:
            
            device:
                This should be an [torch.device] 
                torch.device("cpu") for put the tensor on host (cpu)
                torch.device("cuda:x") for put the tensor on GPU with index x

        Example:

            Construct a tensor (default is on cpu)
            >>> a = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,4)])
            
            Set to GPU.
            >>> a.to(torch.device("cuda:0"))


        """
        if not isinstance(device,torch.device):
            raise TypeError("[ERROR] UniTensor.to()","only support device argument in this version as torch.device")
        self.Storage = self.Storage.to(device)         

    ## print layout:
    def Print_diagram(self):
        """
        This is the beauty print of the tensor diagram. Including the information for the placeing device 
        ::
            1.The left hand side is always the In-bond, and the right hand side is always the Out-bond. 
            2.The number attach to the out-side of each leg is the Bond-dimension. 
            3.The number attach to the in-side of each leg is the label. 
            4.The real memory layout are follow clock-wise from upper-right to upper-left.
                          

            [ex:] Rank = 4. 
            shape: (1,2,3,6) 
            D_IN=[1,2], D_OUT=[3,6], labels=[0,5,3,11]

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
            Compare two UniTensor.
            ::
                a == b

            where a & b are UniTensors.

            Note that this will only compare the shape of Storage. Not the content of torch tensor.


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
        """ 
            Return the shape of UniTensor

            Return:
                torch.Size object, using np.array() or list() to convert to numpy array and python list. 

        """
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

    
    def __truediv__(self,other):
        if isinstance(other, self.__class__):
            return UniTensor(self.D_IN,self.D_OUT,\
                             self.Label_IN,self.Label_OUT,\
                             self.Storage / other.Storage)
        else :
            return UniTensor(self.D_IN,self.D_OUT,\
                             self.Label_IN,self.Label_OUT,\
                             self.Storage / other)
    

    ## This is the same function that behaves as the memberfunction.
    def Svd(self): 
        """ 
            This is the member function of Svd, see Tor10.linalg.Svd() 
        """
        return Svd(self)

    #def Svd_truncate(self):
    #    """ 
    #        This is the member function of Svd_truncate, see Tor10.Svd_truncate() 
    #    """
    #    return Svd_truncate(self)

    def Norm(self):
        """ 
            This is the member function of Norm, see Tor10.linalg.Norm() 
        """
        return Norm(self)

    def Det(self):
        """ 
            This is the member function of Det, see Tor10.linalg.Det() 
        """
        return Det(self)

    def Matmul(self,b):
        """ 
            This is the member function of Matmul, see Tor10.linalg.Matmul() 
        """
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
        """
        Randomize the UniTensor.

            Note that in current version, only a UniTensor without symmetry quantum numbers can be randomized.

        Return: 
            
            self
        """
        if self.bonds[0].qnums is None:
            _Randomize(self)
        else:
            ## Qnum_ipoint
            raise Exception("[Abort] UniTensor.Rand for symm TN is under developing")
        return self

    def CombineBonds(self,labels_to_combine):
        """
        """
        _CombineBonds(self,labels_to_combine)

    def Contiguous(self):
        """
        """
        self.Storage = self.Storage.contiguous()

    def is_contiguous(self):
        """
        """
        return self.Storage.is_contiguous()        


    def Permute(self,maper,N_inbond,by_label=False):
        """
        """
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
        """
        """
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
    def GetTotalQnums(self):
        """
        """
        if self.bonds[0].qnums is None:
            raise TypeError("UniTensor.GetTotalQnums","[ERROR] GetTotal Qnums from a non-symm tensor")
        tmp = np.array([ (x.bondType is BD_OUT) for x in self.bonds])
        maper = np.argsort(tmp)
        tmp_bonds = self.bonds[maper]
        tmp_labels = self.labels[maper]
        Nin = len(tmp[tmp==False])

        if (Nin==0) or (Nin==len(self.bonds)):
            raise Exception("UniTensor.GetTotalQnums","[ERROR] The TN symmetry structure is incorrect, without either any in-bond or any-outbond")

        #virtual_cb-in
        cb_inbonds = copy.deepcopy(tmp_bonds[0])
        cb_inbonds.combine(tmp_bonds[1:Nin])

        cb_outbonds = copy.deepcopy(tmp_bonds[Nin])
        cb_outbonds.combine(tmp_bonds[Nin+1:])

        return cb_inbonds,cb_outbonds


    def PutBlock(self,block,*qnum):
        """
        """
        ## Note, block should be a numpy array.
        if self.bonds[0].qnums is None: 
            raise Exception("[Warning] PutBlock cannot be use for non-symmetry TN. Use SetElem instead.")
        else:
            if len(qnum) != self.bonds[0].nsym :
                raise ValueError("UniTensor.PutBlock","[ERROR] The qnumtum numbers not match the number of type.")

            if self.is_diag:
                raise TypeError("UniTensor.PutBlock","[ERROR] Cannot put block on a diagonal tensor (is_diag=True)")

            ## create a copy of bonds and labels information that has all the BD_IN on first.            
            # [IN, IN, ..., IN, IN, OUT, OUT, ..., OUT, OUT]
            tmp = np.array([ (x.bondType is BD_OUT) for x in self.bonds])
            maper = np.argsort(tmp)
            tmp_bonds = self.bonds[maper]
            tmp_labels = self.labels[maper]
            Nin = len(tmp[tmp==False])
            if (Nin==0) or (Nin==len(self.bonds)):
                raise Exception("UniTensor.PutBlock","[ERROR] Trying to put a block on a TN without either any in-bond or any out-bond")

            #virtual_cb-in
            cb_inbonds = copy.deepcopy(tmp_bonds[0])
            cb_inbonds.combine(tmp_bonds[1:Nin])
            i_in = np.argwhere(cb_inbonds.qnums[:,0]==qnum[0]).flatten()
            for n in np.arange(1,self.bonds[0].nsym,1):
                i_in = np.intersect1d(i_in, np.argwhere(cb_inbonds.qnums[:,n]==qnum[n]).flatten())
            if len(i_in) == 0:
                raise Exception("UniTensor.PutBlock","[ERROR] Trying to put a qnum block that is not exists in the total Qnum of in-bonds in current TN.")

            #virtual_cb_out            
            cb_outbonds = copy.deepcopy(tmp_bonds[Nin])
            cb_outbonds.combine(tmp_bonds[Nin+1:])
            i_out = np.argwhere(cb_outbonds.qnums[:,0]==qnum[0]).flatten()
            for n in np.arange(1,self.bonds[0].nsym,1):
                i_out = np.intersect1d(i_out, np.argwhere(cb_outbonds.qnums[:,n]==qnum[n]).flatten())
            if len(i_out) == 0:
                raise Exception("UniTensor.PutBlock","[ERROR] Trying to put a qnum block that is not exists in the totoal Qnum out-bonds in current TN.")
            
            rev_maper = np.argsort(maper) 
            self.Storage = self.Storage.permute(*maper)
            ori_shape = self.Storage.shape
            print(self.Storage.shape)
            ## this will copy a new tensor , future can provide an shallow copy with no new tensor will create, using .view() possibly handy for Getblock and change the element inplace.
            self.Storage = self.Storage.reshape(np.prod(ori_shape[:Nin]),-1)
            print(self.Storage.shape)
            ## no need to check if the size match. if the size doesn't match, let torch handle the error.
            if isinstance(block,np.ndarray):
                self.Storage[np.ix_(i_in,i_out)] = torch.from_numpy(block).to(torch.float64)
            elif isinstance(block,self.Storage.__class__):
                self.Storage[np.ix_(i_in,i_out)] = block
            else:
                raise TypeError("UniTensor.PutBlock","[ERROR] the block can only be an np.array or a %s"%(self.Storage.__class__))
            
            self.Storage = self.Storage.reshape(*ori_shape).permute(*rev_maper)

    
    def GetBlock(self,*qnum):
        """
        """
        if self.bonds[0].qnums is None:

            if self.is_diag:
                raise TypeError("UniTensor.GetBlock","[ERROR] Cannot get block on a diagonal tensor (is_diag=True)")
            
            print("[Warning] GetBlock a non-symmetry TN will return self regardless of qnum parameter pass in.")
            return self

        else:
            if len(qnum) != self.bonds[0].nsym :
                raise ValueError("UniTensor.GetBlock","[ERROR] The qnumtum numbers not match the number of type.")

            if self.is_diag:
                raise TypeError("UniTensor.GetBlock","[ERROR] Cannot get block on a diagonal tensor (is_diag=True)")
           
            
    
            #######
            ## create a copy of bonds and labels information that has all the BD_IN on first.            
            # [IN, IN, ..., IN, IN, OUT, OUT, ..., OUT, OUT]
            tmp = np.array([ (x.bondType is BD_OUT) for x in self.bonds])
            maper = np.argsort(tmp)
            tmp_bonds = self.bonds[maper]
            tmp_labels = self.labels[maper]
            Nin = len(tmp[tmp==False])
            if (Nin==0) or (Nin==len(self.bonds)):
                raise Exception("UniTensor.GetBlock","[ERROR] Trying to get a block on a TN without either any in-bond or any out-bond")

            #virtual_cb-in
            cb_inbonds = copy.deepcopy(tmp_bonds[0])
            cb_inbonds.combine(tmp_bonds[1:Nin])
            i_in = np.argwhere(cb_inbonds.qnums[:,0]==qnum[0]).flatten()
            for n in np.arange(1,self.bonds[0].nsym,1):
                i_in = np.intersect1d(i_in, np.argwhere(cb_inbonds.qnums[:,n]==qnum[n]).flatten())
            if len(i_in) == 0:
                raise Exception("UniTensor.GetBlock","[ERROR] Trying to get a qnum block that is not exists in the total Qnum of in-bonds in current TN.")

            #virtual_cb_out            
            cb_outbonds = copy.deepcopy(tmp_bonds[Nin])
            cb_outbonds.combine(tmp_bonds[Nin+1:])
            i_out = np.argwhere(cb_outbonds.qnums[:,0]==qnum[0]).flatten()
            for n in np.arange(1,self.bonds[0].nsym,1):
                i_out = np.intersect1d(i_out, np.argwhere(cb_outbonds.qnums[:,n]==qnum[n]).flatten())
            if len(i_out) == 0:
                raise Exception("UniTensor.GetBlock","[ERROR] Trying to get a qnum block that is not exists in the totoal Qnum out-bonds in current TN.")
            
            ## virtual permute:
            rev_maper = np.argsort(maper) 
            self.Storage = self.Storage.permute(*maper)
            ori_shape = self.Storage.shape

            ## this will copy a new tensor , future can provide an shallow copy with no new tensor will create, using .view() possibly handy for Getblock and change the element inplace.
            out = self.Storage.reshape(np.prod(ori_shape[:Nin]),-1)[np.ix_(i_in,i_out)]
            
            self.Storage = self.Storage.permute(*rev_maper)

            #print(out)
            
            return UniTensor(bonds =[Bond(BD_IN,dim=out.shape[0]),Bond(BD_OUT,dim=out.shape[1])],\
                             labels=[1,2],\
                             torch_tensor=out,\
                             check=False)
            

        





###############################################################
#
# Action function 
#
##############################################################
## I/O
def Save(a,filename):
    """
    """
    if not isinstance(filename,str):
        raise TypeError("Save","[ERROR] Invalid filename.")
    if not isinstance(a,UniTensor):
        raise TypeError("Save","[ERROR] input must be the UniTensor")
    f = open(filename,"wb")
    pkl.dump(a,f)
    f.close()

def Load(filename):
    """
    """
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
    """
    """
    if isinstance(a,UniTensor) and isinstance(b,UniTensor):


        ## get same vector:
        same, a_ind, b_ind = np.intersect1d(a.labels,b.labels,return_indices=True)

        ## -v
        #print(a_ind,b_ind)

        if(len(same)):
            ## check dim:
            #for i in range(len(a_ind)):
            #    if a.bonds[a_ind[i]].dim != b.bonds[b_ind[i]].dim:
            #        raise ValueError("Contact(a,b)","[ERROR] contract Bonds that has different dim.")


            ## Qnum_ipoint
            if (a.bonds[0].qnums is not None)^(b.bonds[0].qnums is not None):
                raise Exception("Contract(a,b)","[ERROR] contract Symm TN with non-sym tensor")

            if(a.bonds[0].qnums is not None):
                for i in range(len(a_ind)):
                    if not a.bonds[a_ind[i]].qnums.all() == b.bonds[b_ind[i]].qnums.all():
                        raise ValueError("Contact(a,b)","[ERROR] contract Bonds that has qnums mismatch.")

            aind_no_combine = np.setdiff1d(np.arange(len(a.labels)),a_ind)
            bind_no_combine = np.setdiff1d(np.arange(len(b.labels)),b_ind)

            #print(aind_no_combine,bind_no_combine)
            
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



