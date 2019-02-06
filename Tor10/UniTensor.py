import torch 
import copy,os
import numpy as np
import pickle as pkl
from .Bond import *
from . import linalg 

## Developer Note:
## [KHW]
## Currently trying to add the Symm. 
## A temporary Abort is use to prevent the user to call the un-support operations on a Symmetry tensor. 
##
##  Find "Qnum_ipoint" keyword for the part that need to be modify accrodingly when considering the Qnums feature. 
##

class UniTensor():

    def __init__(self, bonds, labels=None, device=torch.device("cpu"),dtype=torch.float64,torch_tensor=None,check=True, is_diag=False, requires_grad=False, name=""):
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
            requires_grad:
                Activate the autograd function for UniTensor. This is the same as torch.Tensor 
                
            name: 
                This states the name of current UniTensor.      

        Private Args:
        

           ** [Warning] Private Args should not be call directly **


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
    
        if requires_grad:
            self.Storage.requires_grad_(True)


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
        Set the UniTensor to dense matrix. 
            Currently only the diagonal matrix is stored as sparsed form. So it only has effect on UniTensor where is_diag = True
        
        Return:
            self

        Example:

            >>> a = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,3)],is_diag=True)
            >>> print(a.is_diag)
            True
        
            
            >>> a.Todense()
            >>> print(a.is_diag)
            False
            

        """
        if self.is_diag==True:
            self.Storage = torch.diag(self.Storage) 
            self.is_diag=False

        return self

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
        return linalg.Svd(self)

    #def Svd_truncate(self):
    #    """ 
    #        This is the member function of Svd_truncate, see Tor10.Svd_truncate() 
    #    """
    #    return Svd_truncate(self)

    def Norm(self):
        """ 
            This is the member function of Norm, see Tor10.linalg.Norm
        """
        return linalg.Norm(self)

    def Det(self):
        """ 
            This is the member function of Det, see Tor10.linalg.Det
        """
        return linalg.Det(self)

    def Matmul(self,b):
        """ 
            This is the member function of Matmul, see Tor10.linalg.Matmul
        """
        return linalg.Matmul(self,b)

    
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
        This function combines the bonds in input UniTensor [a] by the specified labels [label].
    
        Args:
            
            labels_to_combine: 
                labels that to be combined. It should be a int list / numpy array of the label. All the bonds with specified labels in the current UniTensor  will be combined

        Example:

            1. Combine Bond for an non-symmetric tensor.

            >>> bds_x = [Tor10.Bond(Tor10.BD_IN,5),Tor10.Bond(Tor10.BD_OUT,5),Tor10.Bond(Tor10.BD_OUT,3)]
            >>> x = Tor10.UniTensor(bonds=bds_x, labels=[4,3,5])
            >>> x.Print_diagram()
            tensor Name : 
            tensor Rank : 3
            on device   : cpu
            is_diag     : False
                    ---------------     
                    |             |     
               4 __ | 5         5 |__ 3  
                    |             |     
                    |           3 |__ 5  
                    |             |     
                    ---------------     
            lbl:4 Dim = 5 |
            IN :
            _
            lbl:3 Dim = 5 |
            OUT :
            _
            lbl:5 Dim = 3 |
            OUT :

            >>> x.CombineBonds([4,3])
            >>> x.Print_diagram()
            tensor Name : 
            tensor Rank : 2
            on device   : cpu
            is_diag     : False
                    ---------------     
                    |             |     
                    |           3 |__ 5  
                    |             |     
                    |          25 |__ 3  
                    |             |     
                    ---------------     
            lbl:5 Dim = 3 |
            OUT :
            _
            lbl:3 Dim = 25 |
            OUT :
            _

            
            2. Combine bonds for a Symetric tensor.

                



        """
        _CombineBonds(self,labels_to_combine)

    def Contiguous(self):
        """
        Make the memory to be contiguous. This is the same as pytorch's contiguous(). 
        Because of the Permute does not move the memory, after permute, only the shape of UniTensor is changed, the underlying memory does not change. The UniTensor in this status is called "non-contiguous" tensor.
        When call the Contiguous(), the memory will be moved to match the shape of UniTensor. 
        *Note* Normally, it is not nessary to call contiguous. Most of the linalg function implicity will make the UniTensor contiguous. If one calls a function that requires a contiguous tensor, the error will be issue. Then you know you have to put UniTensor.Contiguous() there.

        Return:
            self

        Example:

            >>> x = Tt.UniTensor(bonds=bds_x, labels=[4,3,5])
            >>> print(x.is_contiguous())
            True

            >>> x.Permute([0,2,1])  
            >>> print(x.is_contiguous())
            False

            >>> x.Contiguous()
            >>> print(x.is_contiguous())
            True
            
        """
        self.Storage = self.Storage.contiguous()
        return self


    def is_contiguous(self):
        """
        Return the status of memory contiguous.

        Return:
            bool, if True, then the Storage of UniTensor is contiguous. if False, then the Storage of UiTensor is non-contiguous. 
 
        """
        return self.Storage.is_contiguous()        


    def Permute(self,maper,N_inbond,by_label=False):
        """
        Permute the bonds of the UniTensor.

        Args:
            maper:
                a python list with integer type elements that the UniTensor permute accroding to. 
            
            N_inbond:
                The number of in-bond after permute.

            by_label:
                bool, when True, the maper using the labels. When False, the maper using the index.

        Example:

            >>> bds_x = [Tt.Bond(Tt.BD_IN,6),Tt.Bond(Tt.BD_OUT,5),Tt.Bond(Tt.BD_OUT,3)]
            >>> x = Tt.UniTensor(bonds=bds_x, labels=[4,3,5])
            >>> y = Tt.UniTensor(bonds=bds_x, labels=[4,3,5])
            >>> x.Print_diagram()
            tensor Name : 
            tensor Rank : 3
            on device   : cpu
            is_diag     : False
                    ---------------     
                    |             |     
                4 __| 6         5 |__ 3  
                    |             |     
                    |           3 |__ 5  
                    |             |     
                    ---------------     
            lbl:4 Dim = 6 |
            IN :
            _
            lbl:3 Dim = 5 |
            OUT :
            _
            lbl:5 Dim = 3 |
            OUT :

            >>> x.Permute([0,2,1],2)
            >>> x.Print_diagram()
            tensor Name : 
            tensor Rank : 3
            on device   : cpu
            is_diag     : False
                    ---------------     
                    |             |     
                4 __| 6         5 |__ 3  
                    |             |     
                5 __| 3           |      
                    |             |     
                    ---------------     
            lbl:4 Dim = 6 |
            IN :
            _
            lbl:5 Dim = 3 |
            IN :
            _
            lbl:3 Dim = 5 |
            OUT :

            >>> y.Permute([3,4,5],2,by_label=True)
            >>> y.Print_diagram()
            tensor Name : 
            tensor Rank : 3
            on device   : cpu
            is_diag     : False
                    ---------------     
                    |             |     
                3 __| 5         3 |__ 5  
                    |             |     
                4 __| 6           |      
                    |             |     
                    ---------------     
            lbl:3 Dim = 5 |
            IN :
            _
            lbl:4 Dim = 6 |
            IN :
            _
            lbl:5 Dim = 3 |
            OUT :


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
        Reshape the UniTensor into the shape specified as [dimer], with the first [N_inbond] Bonds as in-bond and other bonds as out-bond. 
        
        Args:

            dimer:
                The new shape of the UniTensor. This should be a python list. 

            N_inbond:
                The number of in-bond.
            
            new_labels:
                The new labels that will be set for new bonds after reshape. 

        Example:

            >>> bds_x = [Tt.Bond(Tt.BD_IN,6),Tt.Bond(Tt.BD_OUT,5),Tt.Bond(Tt.BD_OUT,3)]
            >>> x = Tt.UniTensor(bonds=bds_x, labels=[4,3,5])
            >>> x.Print_diagram()
            tensor Name : 
            tensor Rank : 3
            on device   : cpu
            is_diag     : False
                    ---------------     
                    |             |     
                4 __| 6         5 |__ 3  
                    |             |     
                    |           3 |__ 5  
                    |             |     
                    ---------------     
            lbl:4 Dim = 6 |
            IN :
            _
            lbl:3 Dim = 5 |
            OUT :
            _
            lbl:5 Dim = 3 |
            OUT :
            

            >>> x.Reshape([2,3,5,3],new_labels=[1,2,3,-1],N_inbond=2)
            >>> x.Print_diagram()
            tensor Name : 
            tensor Rank : 4
            on device   : cpu
            is_diag     : False
                    ---------------     
                    |             |     
                1 __| 2         5 |__ 3  
                    |             |     
                2 __| 3         3 |__ -1 
                    |             |     
                    ---------------     
            lbl:1 Dim = 2 |
            IN :
            _
            lbl:2 Dim = 3 |
            IN :
            _
            lbl:3 Dim = 5 |
            OUT :
            _
            lbl:-1 Dim = 3 |
            OUT :

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
        Return two combined bond objects that has the information for the total qnums at in and out bonds.
        
        Return:
            qnums_inbonds, qnums_outbonds:

            qnums_inbonds:
                a Tor10.Bond, the combined in-bond
            
            qnums_outbonds:
                a Tor10.Bond, the combined out-bond.

                
        Example:

            * Multiple Symmetry::

                ## multiple Qnum:
                ## U1 x U1 x Z2 x Z4
                ## U1 = {-2,-1,0,1,2}
                ## Z2 = {-1,1}
                ## Z4 = {0,1,2,3}
                bd_sym_1 = Tt.Bond(Tt.BD_IN,3,qnums=[[0, 2, 1, 0],
                                                     [1, 1,-1, 1],
                                                     [2,-1, 1, 0]])
                bd_sym_2 = Tt.Bond(Tt.BD_IN,4,qnums=[[-1, 0,-1, 3],
                                                     [ 0, 0,-1, 2],
                                                     [ 1, 0, 1, 0],
                                                     [ 2,-2,-1, 1]])
                bd_sym_3 = Tt.Bond(Tt.BD_OUT,2,qnums=[[-1,-2,-1,2],
                                                      [ 1, 1, -2,3]])

                sym_T = Tt.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],labels=[1,2,3],dtype=tor.float64)
                
            >>> tqin, tqout = sym_T.GetTotalQnums()
            >>> print(tqin)
            Dim = 12 |
            IN  : -1 +0 +1 +2 +0 +1 +2 +3 +1 +2 +3 +4
                  +2 +2 +2 +0 +1 +1 +1 -1 -1 -1 -1 -3
                  +0 +0 +2 +0 -2 -2 +0 -2 +0 +0 +2 +0
                  +3 +2 +0 +1 +4 +3 +1 +2 +3 +2 +0 +1

            >>> print(tqout)
            Dim = 2 |
            OUT : -1 +1
                  -2 +1
                  -1 -2
                  +2 +3
                
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
        Return the Block specify by the quantum number(s). If the UniTensor is non-symmetry, return self. 

        Args:
            *qnum:
                The quantum number(s). Note that when get-block on a High-rank tensor, the quantum number represent the total quantum number of all the in(out)-bonds.

        Return:
            * UniTensor, rank-2 (for symmetry tensor)
            * self (only if the UniTensor is non-symmetry tensor)
        
        Example:
            * Single Symmetry::
                
                bd_sym_1 = Tt.Bond(Tt.BD_IN,3,qnums=[[0],[1],[2]])
                bd_sym_2 = Tt.Bond(Tt.BD_IN,4,qnums=[[-1],[2],[0],[2]])
                bd_sym_3 = Tt.Bond(Tt.BD_OUT,5,qnums=[[4],[2],[-1],[5],[1]])
                sym_T = Tt.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],labels=[10,11,12],dtype=tor.float64)
                
            >>> sym_T.Print_diagram()
            tensor Name : 
            tensor Rank : 3
            on device   : cpu
            is_diag     : False
                    ---------------     
                    |             |     
               10 __| 3         5 |__ 12 
                    |             |     
               11 __| 4           |      
                    |             |     
                    ---------------     
            lbl:10 Dim = 3 |
            IN  : +0 +1 +2
            _
            lbl:11 Dim = 4 |
            IN  : -1 +2 +0 +2
            _
            lbl:12 Dim = 5 |
            OUT : +4 +2 -1 +5 +1

            >>> q_in, q_out = GetTotalQnums()
            >>> print(q_in)
            Dim = 12 |
            IN  : -1 +2 +0 +2 +0 +3 +1 +3 +1 +4 +2 +4
            
            >>> print(q_out)
            Dim = 5 |
            OUT : +4 +2 -1 +5 +1

            >>> block_2 = sym_T.GetBlock(2)
            >>> print(block_2)
            Tensor name: 
            is_diag    : False
            tensor([[0.],
                    [0.],
                    [0.]], dtype=torch.float64)

            
            * Multiple Symmetry::

                ## multiple Qnum:
                ## U1 x U1 x Z2 x Z4
                ## U1 = {-2,-1,0,1,2}
                ## Z2 = {-1,1}
                ## Z4 = {0,1,2,3}
                bd_sym_1 = Tt.Bond(Tt.BD_IN,3,qnums=[[0, 2, 1, 0],
                                                     [1, 1,-1, 1],
                                                     [2,-1, 1, 0]])
                bd_sym_2 = Tt.Bond(Tt.BD_IN,4,qnums=[[-1, 0,-1, 3],
                                                     [ 0, 0,-1, 2],
                                                     [ 1, 0, 1, 0],
                                                     [ 2,-2,-1, 1]])
                bd_sym_3 = Tt.Bond(Tt.BD_OUT,2,qnums=[[-1,-2,-1,2],
                                                      [ 1, 1, -2,3]])

                sym_T = Tt.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],labels=[1,2,3],dtype=tor.float64)
                
            >>> tqin, tqout = sym_T.GetTotalQnums()
            >>> print(tqin)
            Dim = 12 |
            IN  : -1 +0 +1 +2 +0 +1 +2 +3 +1 +2 +3 +4
                  +2 +2 +2 +0 +1 +1 +1 -1 -1 -1 -1 -3
                  +0 +0 +2 +0 -2 -2 +0 -2 +0 +0 +2 +0
                  +3 +2 +0 +1 +4 +3 +1 +2 +3 +2 +0 +1

            >>> print(tqout)
            Dim = 2 |
            OUT : -1 +1
                  -2 +1
                  -1 -2
                  +2 +3

            >>> block_1123 = sym_T.GetBlock(1,1,-2,3)
            >>> print(block_1123)
            Tensor name: 
            is_diag    : False
            tensor([[0.]], dtype=torch.float64)


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
            

    ## Autograd feature: 
    def requires_grad(self,is_grad=None):
        """ 
        The status for the autograd property.

        Args:
            is_grad: 
                bool, if the autograd mechanism should be activate on this UniTensor. 
                If the argument is not set, it will return the current autograd status. 
                
        Return:
            bool, return only when is_grad argument is ignored. 

        Example:
        ::
            bds_x = [Tt.Bond(Tt.BD_IN,5),Tt.Bond(Tt.BD_OUT,5),Tt.Bond(Tt.BD_OUT,3)]
            x = Tt.UniTensor(bonds=bds_x, labels=[4,3,5])

    
        >>> print(x.requires_grad())
        False

        >>> x.requires_grad(True)
        >>> print(x.requires_grad())
        True

        >>> x.requires_grad(False)
        >>> print(x.requires_grad())
        False

        
        """
        if is_grad is None:
            return self.Storage.requires_grad
        else:
            self.Storage.requires_grad_(bool(is_grad))


    def grad(self):
        """
        Return the gradient tensors subject to x where x is the current UniTensor. The return is None by default and becomes a UniTensor the first time a call backward(). The future calls to backward() will accumulate (add) gradient into it.

        This is the same as torch.Tensor.grad
   

        :math:`d/dx`

        Return: 
            UniTensor, the shape of the return UniTensor and it's bonds are the same as the original UniTensor, but with default labels.

        Example:
        
            >>> x = Tor10.UniTensor(bonds=[Tor10.Bond(BD_IN,2),Tor10.Bond(BD_OUT,2)],requires_grad=True)
            >>> print(x)
            Tensor name: 
            is_diag    : False
            tensor([[0., 0.],
                    [0., 0.]], dtype=torch.float64, requires_grad=True)

            >>> y = (x + 4)**2
            >>> print(y)
            Tensor name: 
            is_diag    : False
            tensor([[16., 16.],
                    [16., 16.]], dtype=torch.float64, grad_fn=<PowBackward0>)

            >>> out = Tor10.Mean(y)
            >>> print(out)
            Tensor name: 
            is_diag    : False
            tensor(16., dtype=torch.float64, grad_fn=<MeanBackward1>)

            >>> out.backward()
            >>> print(x.grad)
            Tensor name: 
            is_diag    : False
            tensor([[2., 2.],
                    [2., 2.]], dtype=torch.float64)

        """
        if self.Storage.grad is None:
            return None
        else:
            return UniTensor(bonds=copy.deepcopy(self.bonds),\
                             torch_tensor=self.Storage.grad,\
                             check=False)
    
    def backward(self):
        """
        Backward the gradient flow in the contructed autograd graph. This is the same as torch.Tensor.backward
        """
        self.Storage.backward()


    def detach(self):
        """
        Detach the current tensor from the current graph, making it a leaf. This is the same as torch.Tensor.detach_()

        Return:
            self
        """
        self.Storage.detach_()
        return self



###############################################################
#
# Action function 
#
##############################################################
## I/O
def Save(a,filename):
    """
    Save a UniTensor to the file

    Args:
        a: 
            The UniTensor that to be saved.
        
        filename:
            The saved file path

    Example:
    ::
        a = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,4)])
        Tor10.Save(a,"a.uniT")
    
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
    Load a UniTensor from the file.

    Args:
        filename: 
            The path of the file to be loaded

    Return:
        UniTensor

    Example:
    ::
        a = Tor10.Load("a.uniT")

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

def Contract(a,b,inbond_first=True):
    """
    Contract two tensors with the same labels. 

    1. two tensors must be the same type, if "a" is a symmetry tensor, "b" must also be a symmetry tensor.
    2. When contract two symmetry tensor, the bonds that to be contracted must have the same qnums.

    Args:
        a:
            UniTensor

        b:
            UniTensor

        inbond_first:
            bool

            * if True , the order of the bonds for the return tensor will be permuted to all the in-bond appears first, then the out-bond.
            * If False, the order of the bonds for the return tensor will have all the remaining bonds of tensor "a" appears first, then the remaining bonds of tensor "b". 
            * This is especially efficient in the case where the in/out bond are not important. By setting inbond_first=False, no additional permute will be perform at the last stage. 

    Return:
        UniTensor

    Example:
    ::
        x = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,5),Tt.Bond(Tt.BD_OUT,5),Tt.Bond(Tt.BD_OUT,4)], labels=[4,3,5])
        y = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,3),Tt.Bond(Tt.BD_OUT,4)],labels=[1,5])


    >>> x.Print_diagram()
    tensor Name : 
    tensor Rank : 3
    on device   : cpu
    is_diag     : False
            ---------------     
            |             |     
        4 __| 5         5 |__ 3  
            |             |     
            |           4 |__ 5  
            |             |     
            ---------------     
    lbl:4 Dim = 5 |
    IN  :
    _
    lbl:3 Dim = 5 |
    OUT :
    _
    lbl:5 Dim = 4 |
    OUT :

    >>> y.Print_diagram()
    tensor Name : 
    tensor Rank : 2
    on device   : cpu
    is_diag     : False
            ---------------     
            |             |     
        1 __| 3         4 |__ 5  
            |             |     
            ---------------     
    lbl:1 Dim = 3 |
    IN  :
    _
    lbl:5 Dim = 4 |
    OUT :

    >>> c = Tt.Contract(x,y)
    >>> c.Print_diagram()
    tensor Name : 
    tensor Rank : 3
    on device   : cpu
    is_diag     : False
            ---------------     
            |             |     
        4 __| 5         5 |__ 3  
            |             |     
        1 __| 3           |      
            |             |     
            ---------------     
    lbl:4 Dim = 5 |
    IN  :
    _
    lbl:1 Dim = 3 |
    IN  :
    _
    lbl:3 Dim = 5 |
    OUT :

    >>> c= Tt.Contract(x,y,inbond_first=False)
    >>> c.Print_diagram()
    tensor Name : 
    tensor Rank : 3
    on device   : cpu
    is_diag     : False
            ---------------     
            |             |     
        4 __| 5         3 |__ 1  
            |             |     
        3 __| 5           |      
            |             |     
            ---------------     
    lbl:4 Dim = 5 |
    IN  :
    _
    lbl:3 Dim = 5 |
    OUT :
    _
    lbl:1 Dim = 3 |
    IN  :


    """
    if isinstance(a,UniTensor) and isinstance(b,UniTensor):

        ## get same vector:
        same, a_ind, b_ind = np.intersect1d(a.labels,b.labels,return_indices=True)


        if(len(same)):

            ## Qnum_ipoint
            if (a.bonds[0].qnums is not None)^(b.bonds[0].qnums is not None):
                raise Exception("Contract(a,b)","[ERROR] contract Symm TN with non-sym tensor")

            if(a.bonds[0].qnums is not None):
                for i in range(len(a_ind)):
                    if not a.bonds[a_ind[i]].qnums.all() == b.bonds[b_ind[i]].qnums.all():
                        raise ValueError("Contact(a,b)","[ERROR] contract Bonds that has qnums mismatch.")

            aind_no_combine = np.setdiff1d(np.arange(len(a.labels)),a_ind)
            bind_no_combine = np.setdiff1d(np.arange(len(b.labels)),b_ind)

            if a.is_diag :
                tmpa = torch.diag(a.Storage).to(a.Storage.device)
            else:   
                tmpa = a.Storage
            
            if b.is_diag :
                tmpb = torch.diag(b.Storage).to(b.Storage.device)
            else:   
                tmpb = b.Storage

            tmp = torch.tensordot(tmpa,tmpb,dims=(a_ind.tolist(),b_ind.tolist()))
            
            new_bonds = np.concatenate([copy.deepcopy(a.bonds[aind_no_combine]),copy.deepcopy(b.bonds[bind_no_combine])])
            
            new_labels = np.concatenate([copy.copy(a.labels[aind_no_combine]),copy.copy(b.labels[bind_no_combine])])
            
            if inbond_first:
                if len(new_bonds)>0:
                    maper = np.argsort([x.bondType==BD_OUT for x in new_bonds])
                    new_bonds = new_bonds[maper]
                    new_labels= new_labels[maper]
                    tmp = tmp.permute(*maper)

            return UniTensor(bonds =new_bonds,\
                             labels=new_labels,\
                             torch_tensor=tmp,\
                             check=False)

        else:
            ## direct product
            
            if a.is_diag :
                tmpa = torch.diag(a.Storage)
            else:   
                tmpa = a.Storage
            
            if b.is_diag :
                tmpb = torch.diag(b.Storage)
            else:   
                tmpb = b.Storage

            tmp = torch.tensordot(tmpa,tmpb,dims=0)
            new_bonds = np.concatenate([copy.deepcopy(a.bonds),copy.deepcopy(b.bonds)])
            new_labels = np.concatenate([copy.copy(a.labels), copy.copy(b.labels)])

            if inbond_first:
                if len(new_bonds)>0:
                    maper = np.argsort([x.bondType==BD_OUT for x in new_bonds])
                    new_bonds = new_bonds[maper]
                    new_labels= new_labels[maper]
                    tmp = tmp.permute(*maper)

            return UniTensor(bonds=new_bonds,\
                             labels=new_labels,\
                             torch_tensor=tmp,\
                             check=False)
            
    else:
        raise Exception('Contract(a,b)', "[ERROR] a and b both have to be UniTensor")


#def Contract_old(a,b):
#    """
#    """
#    if isinstance(a,UniTensor) and isinstance(b,UniTensor):
#
#
#        ## get same vector:
#        same, a_ind, b_ind = np.intersect1d(a.labels,b.labels,return_indices=True)
#
#        ## -v
#        #print(a_ind,b_ind)
#
#        if(len(same)):
#            ## check dim:
#            #for i in range(len(a_ind)):
#            #    if a.bonds[a_ind[i]].dim != b.bonds[b_ind[i]].dim:
#            #        raise ValueError("Contact(a,b)","[ERROR] contract Bonds that has different dim.")
#
#
#            ## Qnum_ipoint
#            if (a.bonds[0].qnums is not None)^(b.bonds[0].qnums is not None):
#                raise Exception("Contract(a,b)","[ERROR] contract Symm TN with non-sym tensor")
#
#            if(a.bonds[0].qnums is not None):
#                for i in range(len(a_ind)):
#                    if not a.bonds[a_ind[i]].qnums.all() == b.bonds[b_ind[i]].qnums.all():
#                        raise ValueError("Contact(a,b)","[ERROR] contract Bonds that has qnums mismatch.")
#
#            aind_no_combine = np.setdiff1d(np.arange(len(a.labels)),a_ind)
#            bind_no_combine = np.setdiff1d(np.arange(len(b.labels)),b_ind)
#
#            #print(aind_no_combine,bind_no_combine)
#            
#            maper_a = np.concatenate([aind_no_combine,a_ind])
#            maper_b = np.concatenate([b_ind,bind_no_combine])
#
#            old_shape = np.array(a.Storage.shape) if a.is_diag==False else np.array([a.Storage.shape[0],a.Storage.shape[0]])
#            combined_dim = np.prod(old_shape[a_ind])
#
#            if a.is_diag :
#                tmpa = torch.diag(a.Storage).to(a.Storage.device)
#            else:   
#                tmpa = a.Storage
#            
#            if b.is_diag :
#                tmpb = torch.diag(b.Storage).to(b.Storage.device)
#            else:   
#                tmpb = b.Storage
#
#            tmp = torch.matmul(tmpa.permute(maper_a.tolist()).reshape(-1,combined_dim),\
#                               tmpb.permute(maper_b.tolist()).reshape(combined_dim,-1))
#            new_shape = [ bd.dim for bd in a.bonds[aind_no_combine]] + [ bd.dim for bd in b.bonds[bind_no_combine]]
#            return UniTensor(bonds =np.concatenate([a.bonds[aind_no_combine],b.bonds[bind_no_combine]]),\
#                             labels=np.concatenate([a.labels[aind_no_combine],b.labels[bind_no_combine]]),\
#                             torch_tensor=tmp.view(new_shape),\
#                             check=False)
#
#        else:
#            ## direct product
#            Nin_a = len([1 for i in range(len(a.labels)) if a.bonds[i].bondType is BD_IN])
#            Nin_b = len([1 for i in range(len(b.labels)) if b.bonds[i].bondType is BD_IN])
#            Nout_a = len(a.labels) - Nin_a
#            Nout_b = len(b.labels) - Nin_b
#
#            new_label = np.concatenate([a.labels, b.labels])
#            DALL = [a.bonds[i].dim for i in range(len(a.bonds))] + [b.bonds[i].dim for i in range(len(b.bonds))]
#
#            maper = np.concatenate([np.arange(Nin_a), len(a.labels) + np.arange(Nin_b), np.arange(Nout_a) + Nin_a, len(a.labels) + Nin_b + np.arange(Nout_b)])
#
#            if a.is_diag :
#                tmpa = torch.diag(a.Storage)
#            else:   
#                tmpa = a.Storage
#            
#            if b.is_diag :
#                tmpb = torch.diag(b.Storage)
#            else:   
#                tmpb = b.Storage
#
#
#            return UniTensor(bonds=np.concatenate([a.bonds[:Nin_a],b.bonds[:Nin_b],a.bonds[Nin_a:],b.bonds[Nin_b:]]),\
#                            labels=np.concatenate([a.labels[:Nin_a], b.labels[:Nin_b], a.labels[Nin_a:], b.labels[Nin_b:]]),\
#                            torch_tensor=torch.ger(tmpa.view(-1),tmpb.view(-1)).reshape(DALL).permute(maper.tolist()),\
#                            check=False)
#            
#    else:
#        raise Exception('Contract(a,b)', "[ERROR] a and b both have to be UniTensor")





## The functions that start with "_" are the private functions

def _CombineBonds(a,label):    
    """
    This function combines the bonds in input UniTensor [a] by the specified labels [label]. The bondType of the combined bonds will always follows the same bondType of bond in [a] with label of the first element in [label] 
    
    Args:
        
        a: 
            UniTensor
        
        label: 

            labels that to be combined. It should be a int list / numpy array of the label. All the bonds with specified labels in the current UniTensor  will be combined

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


def From_torch(torch_tensor,N_inbond,labels=None):
    """ 
    Construct UniTensor from torch.Tensor. 
    
    If the input torch_tensor belongs to a autograd graph, the contructed UniTensor will preserve the role of the input torch_tensor in the computational graph.

    Args:
        torch_tensor:
            Torch.Tensor
    
        N_inbond:
            int, The number of inbond. Note that the first [N_inbond] bonds will be set to Tor10.BD_IN, and the remaining bonds will be set to Tor10.BD_OUT
        
        labels:
            python list or 1d numpy array, The labels for each bonds. If ignore, the constucted UniTensor will using the default labels for each bond.

    Return:
        UniTensor

    Example:
    
        >>> x = torch.ones(3,3)
        >>> print(x)
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]])

        >>> y = Tt.From_torch(x,N_inbond=1,labels=[4,5])
        >>> y.Print_diagram()
        tensor Name : 
        tensor Rank : 2
        on device   : cpu
        is_diag     : False
                ---------------     
                |             |     
            4 __| 3         3 |__ 5  
                |             |     
                ---------------     
        lbl:4 Dim = 3 |
        IN  :
        _
        lbl:5 Dim = 3 |
        OUT :

        >>> print(y)
        Tensor name: 
        is_diag    : False
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]])


        >>> x2 = torch.ones(3,4,requires_grad=True)
        >>> print(x2)
        tensor([[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]], requires_grad=True)
        
        >>> y2 = Tt.From_torch(x2,N_inbond=1)
        >>> print(y2.requires_grad())
        True


    """
    if not isinstance(torch_tensor,torch.Tensor):
        raise TypeError("From_torch","[ERROR] can only accept torch.Tensor")

    shape = torch_tensor.shape
    
    if N_inbond > len(shape):
        raise ValueError("From_torch","[ERROR] N_inbond exceed the rank of input torch tensor.")

    new_bonds = [Bond(BD_IN,shape[i]) for i in range(N_inbond)]+\
                [Bond(BD_OUT,shape[i]) for i in np.arange(N_inbond,len(shape),1)]


    return UniTensor(bonds=new_bonds,labels=labels,torch_tensor=torch_tensor)




