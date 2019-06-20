## [DEBUG] >>> 
## Note, set this to True to enable debug section
DEBUG = False
## <<<


import os
import pickle as pkl

import torch

from . import linalg
from .Bond import *
from .Bond import _fx_GetCommRows


## Developer Note:
## [KHW]
## from v0.3+, we deprecate dense Symmetry. 
## Using a is_symm as master switch. 
## Find "[Fusion tree]" keyword for future extend of non-abelian / fermion etc. 
## Find "DEBUG" keywork to comment the debug section when in release!!. 

def _fx_decompress_idx(x, accu_offsets):
    y = []
    for i in range(len(accu_offsets)):
        y.append(np.array(x / accu_offsets[i]).astype(np.int))
        x = x % accu_offsets[i]
    return np.array(y).swapaxes(0, 1)


class UniTensor:

    def _mac(self, torch_tensor=None, braket=None, sym_mappers=None):
        """
        Memory Allocation and Check (_mac)

            torch_tensor :
                This is the internal arguments in current version. It should not be directly use, otherwise may cause inconsistence with Bonds and memory layout.
                    *For Developer:
                        > The torch_tensor should have the same rank as len(label), and with each bond dimensions strictly the same as describe as in bond in self.bonds.

            check :
                If False, all the checking across bonds/labels/Storage.shape will be ignore.

            braket :
                If set, the braket -1 or +1 indicate the bond are BD_KET or BD_BRA.
                It is handy for calculating reverse quantum flow / blocks when bra-bond is permuted to col-space
                (unmatched braket)

            sym_mappers:
                A tuple, used to pass the shallow permute informations / block mapping information.
        """
        if braket is not None:
            self.braket = copy.deepcopy(braket)
            self._check_braket()

        if sym_mappers is not None:
            self._mapper = copy.deepcopy(sym_mappers[0])
            self._inv_mapper = copy.deepcopy(sym_mappers[1])

            self._Ket_mapper_blks = copy.deepcopy(sym_mappers[2])
            self._Ket_invmapper_blks = copy.deepcopy(sym_mappers[3])
            self._Bra_mapper_blks = copy.deepcopy(sym_mappers[4])
            self._Bra_invmapper_blks = copy.deepcopy(sym_mappers[5])
            self._contiguous = copy.deepcopy(sym_mappers[6])
            self._accu_off_in = copy.deepcopy(sym_mappers[7])
            self._accu_off_out = copy.deepcopy(sym_mappers[8])
            self._block_qnums = copy.deepcopy(sym_mappers[9])
            # if torch_tensor is None:
            #    raise TypeError("UniTensor.__init__","[ERROR], pass the interface must accompany with torch_tensor")

        if torch_tensor is not None:
            self.Storage = torch_tensor

    def __init__(self, bonds, rowrank=None, labels=None, device=torch.device("cpu"), dtype=torch.float64, is_diag=False,
                 requires_grad=False, name="", check=True):
        """
        This is the constructor of UniTensor.

        Public Args:

            bonds:
                List of bonds.
                It should be an list or np.ndarray with len(list) being the number of bonds.

            rowrank:
                The number of bonds in row-space.
                The first [rowrank] bonds will be define as the row-space (which means the row space when flatten as Matrix), and the other bonds will be defined as in the col-space (which is the column space when flatten as Matrix).
                When interprete the memory layout as Matrix, the combine of first rowrank bonds will be the row and the other bond will be column.


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
                Note that if is_diag=True, then the UniTensor is strictly required to be a square rank-2 tensor.

            requires_grad:
                Activate the autograd function for UniTensor. This is the same as torch.Tensor

            name:
                This states the name of current UniTensor.

                
        Example for how to create a UniTensor:

            * create a rank-2 untagged UniTensor (matrix) with shape (3,4):
            >>> a = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(4)],rowrank=1)
            >>> a.Print_diagram(bond_info=True)
            -----------------------
            tensor Name : 
            tensor Rank : 2
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 0 ____| 3         4 |____ 1  
                       \             /     
                        -------------      
            lbl:0 Dim = 3 |
            REG     :
            _
            lbl:1 Dim = 4 |
            REG     :

            * create a rank-3 untagged UniTensor with one bond in row-space and two bonds in col-space, shape (3,4,5) and set labels [-3,4,1] for each bond:
            >>> c = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(4),tor10.Bond(5)],rowrank=1,labels=[-3,4,1])
            >>> c.Print_diagram(bond_info=True)
            tensor Name : 
            tensor Rank : 3
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                -3 ____| 3         4 |____ 4  
                       |             |     
                       |           5 |____ 1  
                       \             /     
                        -------------      
            lbl:-3 Dim = 3 |
            REG     :
            _
            lbl:4 Dim = 4 |
            REG     :
            _
            lbl:1 Dim = 5 |
            REG     :


            * create a rank-0 UniTensor
            >>> rk0t = tor10.UniTensor(bonds=[])
            >>> rk0t.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 0
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       \             /     
                        -------------  

            >>> print(rk0t)
            Tensor name: 
            is_diag    : False
            tensor(0., dtype=torch.float64) 

            * create a rank-3 tagged UniTensor with two bonds in row-space and two bonds in col-space, shape (2,3,4,5)
            >>> bds  = [tor10.Bond(2,tor10.BD_KET),tor10.Bond(3,tor10.BD_KET),tor10.Bond(4,tor10.BD_BRA),tor10.Bond(5,tor10.BD_BRA)]
            >>> o = tor10.UniTensor(bonds=bds,rowrank=2)
            >>> o.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 4
            has_symmetry: False
            on device     : cpu
            is_diag       : False
            braket_form : True
                  |ket>               <bra| 
                       ---------------      
                       |             |     
                 0 > __| 2         4 |__ < 2  
                       |             |     
                 1 > __| 3         5 |__ < 3  
                       |             |     
                       ---------------  


            * note that if the BRA bond is not in the col-space, or KET bond is not in the row-space, the tensor is in the so called "non-braket_form, which will have a * symbol indicating the mismatch."
            >>> bd2 = [tor10.Bond(2,tor10.BD_KET),tor10.Bond(5,tor10.BD_BRA),tor10.Bond(4,tor10.BD_BRA),tor10.Bond(3,tor10.BD_KET)]
            >>> c_mismatch = tor10.UniTensor(bonds=bd2,rowrank=2)
            >>> c_mismatch.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 4
            has_symmetry: False
            on device     : cpu
            is_diag       : False
            braket_form : False
                  |ket>               <bra| 
                       ---------------      
                       |             |     
                 0 > __| 2         4 |__ < 2  
                       |             |     
                 1 <*__| 5         3 |__*> 3  
                       |             |     
                       ---------------  


            * create a rank-2 UniTensor with one inbond, one outbond, shape (3,4) on GPU-0:
            >>> d = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(4)],rowrank=1,device=torch.device("cuda:0"))

            * create a diagonal 6x6 rank-2 tensor(matrix):
              Note that if is_diag is True, rowrank must be 1.
            >>> e = tor10.UniTensor(bonds=[tor10.Bond(6),tor10.Bond(6)],rowrank=1,is_diag=True)

            Note that when is_diag is set to True, the UniTensor should be a square matrix.

            * create a rank-3 UniTensor with two bonds in row-space and one bond in col-space, and single precision:
            >>> f = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(4),tor10.Bond(5)],rowrank=2,labels=[-3,4,1],dtype=torch.float32)

            * create a rank-3 UniTensor with U1 symmetry:
            >>> bd_sym_1 = tor10.Bond(3,tor10.BD_KET,qnums=[[0],[1],[2]])
            >>> bd_sym_2 = tor10.Bond(4,tor10.BD_KET,qnums=[[-1],[2],[0],[2]])
            >>> bd_sym_3 = tor10.Bond(5,tor10.BD_BRA,qnums=[[4],[2],[-1],[5],[1]])
            >>> symT = tor10.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],rowrank=2,labels=[10,11,12])
            >>> symT.Print_diagram(bond_info=True)
            -----------------------
            tensor Name : 
            tensor Rank : 3
            has_symmetry: True
            on device     : cpu
            braket_form : True
                  |ket>               <bra| 
                       ---------------      
                       |             |     
                10 > __| 3         5 |__ < 12 
                       |             |     
                11 > __| 4           |        
                       |             |     
                       ---------------    
            lbl:10 Dim = 3 |
            KET     : U1::  +2 +1 +0
            _
            lbl:11 Dim = 4 |
            KET     : U1::  +2 +2 +0 -1
            _        
            lbl:12 Dim = 5 |
            BRA     : U1::  +5 +4 +2 +1 -1


        """

        ## general property:---------------------------------
        self.name = name

        ## bonds:
        self.bonds = np.array([copy.deepcopy(bonds[i]) for i in range(len(bonds))])

        # labels: 
        if labels is None:
            if len(self.bonds) == 0:
                self.labels = np.array([], dtype=np.int)
            else:
                self.labels = np.arange(len(self.bonds))
        else:
            self.labels = np.array(copy.deepcopy(labels), dtype=np.int)

        ## checking :
        if check:
            # check # of labels consist with bond.
            if not len(self.labels) == (len(self.bonds)):
                raise Exception("UniTensor.__init__", "labels size is not consistence with the rank")
            # Bonds:
            if rowrank is not None:
                if rowrank < 0 or rowrank > len(self.bonds):
                    raise Exception("UniTensor.__init__", "the rowrank should be >=0 and < # of bonds")

            if len(self.bonds) != 0:

                ## check duplicate label
                if not len(np.unique(self.labels)) == len(self.labels):
                    raise Exception("UniTensor.__init__", "labels contain duplicate element.")

                ## check qnums:
                isSymm = np.unique([(bd.qnums is None) for bd in self.bonds])
                if len(isSymm) != 1:
                    raise TypeError("UniTensor.__init__",
                                    "the bonds are not consistent. Cannot have mixing bonds of with and without symmetry (qnums).")
            else:
                if is_diag:
                    raise Exception("UniTensor.__init__", "the scalar tensor (rank-0) cannot have is_diag=True.")

        ## braket, is_braket:
        # is_tag = False if len(self.bonds)==0 else (self.bonds[0].bondType != BD_REG)
        self.is_braket = None
        self.braket = None
        self.rowrank = rowrank

        if check:
            if len(self.bonds) != 0:
                if self.bonds[0].bondType != BD_REG:
                    self.braket = np.array([BondType[self.bonds[i].bondType] for i in range(len(self.bonds))],
                                           dtype=np.int)

            if self.rowrank is None:
                if len(self.bonds) == 0:
                    self.rowrank = 0
                else:
                    if self.braket is not None:
                        self.rowrank = len(np.argwhere(self.braket == BondType[BD_KET]))
                    else:
                        raise Exception(
                            "[ERROR] for UniTensor init with all the bond are regular, rowrank should be provided")
            else:
                self.rowrank = int(rowrank)

            self._check_braket()

        ## check is_symm:
        self.is_symm = False if len(self.bonds) == 0 else (self.bonds[0].qnums is not None)
        self.is_diag = is_diag

        if not self.is_symm:
            ## non-symmetry properties:----------------------------
            self.is_diag = is_diag
            if check:
                if is_diag:
                    if not len(self.labels) == 2:
                        raise TypeError("UniTensor.__init__", "is_diag=True require Tensor rank==2")

                    if not self.rowrank == 1:
                        raise TypeError("UniTensor.__init__",
                                        "is_diag=True require Tensor rank==2, with 1 inbond and 1 outbond (rowrank=1)")

                    if not self.bonds[0].dim == self.bonds[1].dim:
                        raise TypeError("UniTensor.__init__", "is_diag=True require Tensor to be square rank-2")

                if self.is_diag:
                    self.Storage = torch.zeros(self.bonds[0].dim, device=device, dtype=dtype)
                else:
                    if len(self.bonds) != 0:
                        DALL = [self.bonds[i].dim for i in range(len(self.bonds))]
                        self.Storage = torch.zeros(tuple(DALL), device=device, dtype=dtype)
                        del DALL
                    else:
                        self.Storage = torch.tensor(0, device=device, dtype=dtype)

            # self.Storage = torch_tensor

        else:
            ## Symmetry properties-------------------------------:
            if check:
                if self.bonds[0].qnums is not None:
                    if len(np.unique([bd.nsym for bd in self.bonds])) != 1:
                        raise TypeError("UniTensor.__init__",
                                        "the number of symmetry type for symmetry bonds doesn't match.")
                if self.rowrank < 1 or self.rowrank >= len(self.bonds):
                    raise TypeError("UniTensor.__init__",
                                    "[ERROR] tensor with symmetry must have at least one rank for row space and one rank for column space")

                nket = len(np.argwhere(self.braket == BondType[BD_BRA]).flatten())
                if nket < 1 or nket >= len(self.bonds):
                    raise TypeError("UniTensor.__init__",
                                    "[ERROR] tensor with symmetry must have at least one bra-bond and one ket-bond")

            ## only activate when symmetry is on.
            self._Ket_mapper_blks = None  ## this follow memory
            self._Bra_mapper_blks = None  ## this follow memory
            self._Ket_invmapper_blks = None  ## this follow memory
            self._Bra_invmapper_blks = None  ## this follow memory
            self._mapper = None  ## memory idx to real idx
            self._inv_mapper = None  ## real idx to memory index
            self._contiguous = True
            self._accu_off_in = None  ## this follows memory
            self._accu_off_out = None  ## this follows memory
            self._block_qnums = None  ## this follows real Tensor, not memory!!!

            ## memory contiguous mapper this 
            if check:

                # calc offsets
                accu_off = []
                tmp = 1
                for i in range(len(self.bonds)):
                    accu_off.append(tmp)
                    tmp *= self.bonds[-1 - i].dim
                accu_off = np.array(accu_off[::-1])
                self._accu_off_in = (accu_off[:self.rowrank] / accu_off[self.rowrank - 1]).astype(np.int)
                self._accu_off_out = accu_off[self.rowrank:]
                del accu_off

                ## mapper 
                self._mapper = np.arange(len(self.bonds)).astype(np.int)
                self._inv_mapper = copy.copy(self._mapper)

                ## Get common qnums for in and out b
                b_tqin, b_tqout = self.GetTotalQnums(physical=False)
                tqin_uni = b_tqin.GetUniqueQnums()
                tqout_uni = b_tqout.GetUniqueQnums()
                C = _fx_GetCommRows(tqin_uni, tqout_uni)
                if len(C.flatten()) == 0:
                    raise TypeError("UniTensor.__init__",
                                    "[ERROR] no vaild block in current Tensor. please check total qnums in total bra/ket bonds have at least one same set of qnums.")

                self.Storage = []
                self._Ket_invmapper_blks = []
                self._Bra_invmapper_blks = []
                self._Ket_mapper_blks = -np.ones((b_tqin.dim, 2)).astype(np.int)
                self._Bra_mapper_blks = -np.ones((b_tqout.dim, 2)).astype(np.int)
                self._block_qnums = []

                for b in range(len(C)):
                    comm = tuple(C[b])
                    idx_in = np.argwhere((b_tqin.qnums == comm).all(axis=1)).flatten()
                    idx_out = np.argwhere((b_tqout.qnums == comm).all(axis=1)).flatten()
                    self.Storage.append(torch.zeros((len(idx_in), len(idx_out)), device=device, dtype=dtype))

                    ## interface
                    self._Ket_invmapper_blks.append(_fx_decompress_idx(idx_in, self._accu_off_in))
                    self._Ket_mapper_blks[idx_in, 0] = b
                    self._Ket_mapper_blks[idx_in, 1] = np.arange(len(idx_in)).astype(np.int)

                    ## interface
                    self._Bra_invmapper_blks.append(_fx_decompress_idx(idx_out, self._accu_off_out))
                    self._Bra_mapper_blks[idx_out, 0] = b
                    self._Bra_mapper_blks[idx_out, 1] = np.arange(len(idx_out)).astype(np.int)
                self._block_qnums = C

        if check:
            if requires_grad:
                self.requires_grad(True)

    def tag_braket(self, tags=None):
        if self.braket is None:
            if tags is None:
                self.braket = []
                for b_in in range(self.rowrank):
                    self.bonds[b_in].bondType = BD_KET
                    self.braket.append(BondType[BD_KET])
                for b_out in range(len(self.bonds) - self.rowrank):
                    self.bonds[b_out + self.rowrank].bondType = BD_BRA
                    self.braket.append(BondType[BD_BRA])
            else:
                # check:
                if len(tags) != len(self.bonds):
                    raise ValueError("[ERROR] tags must match the rank of bonds.")
                if all([x == BD_REG for x in rags]):
                    raise ValueError("[ERROR] tags cannot contain BD_REG")
                for b in range(len(self.bonds)):
                    self.bonds[b].bondType = tags[b]
                    self.braket.append(BondType[tags[b]])

            self.braket = np.array(self.braket)
            self.is_braket = True

    def untag_braket(self):
        if self.is_symm:
            raise Exception("[ERROR]", "Cannot untag bra/ket on the bonds for symmetry tensor.")

        if self.braket is None:
            pass

        else:
            self.is_braket = None
            self.braket = None
            for b in range(len(self.bonds)):
                self.bonds[b].bondType = BD_REG

    def _check_braket(self):
        """
            This is internal function!!
        """
        if self.braket is not None:
            if (self.braket[:self.rowrank] == BondType[BD_KET]).all() and (
                    self.braket[self.rowrank:] == BondType[BD_BRA]).all():
                self.is_braket = True
            else:
                self.is_braket = False

    def is_braket_form(self):
        """ 
        Return if the current tensor is in braket_form. It can only be called on a tagged UniTensor
        (with or without symmetries)

        Return:

            bool.     
                

        """
        if self.braket is None:
            raise Exception("[ERROR] for a tensor with regular bonds, there is no property of barket.")

        return self.is_braket

    def braket_form(self):
        """
        Permute the UniTensor to bra-ket form. 

        [Tech.Note] the permuted UniTensor can be non-contiguous depending on the underlying memory layout. 

        Return :
            self

        """
        if self.braket is None:
            raise Exception("[ERROR] for a tensor with regular bonds, there is no property of barket.")
        x = np.argsort(self.braket)
        Nin = len(np.argwhere(self.braket == BondType[BD_KET]))
        self.Permute(x, rowrank=Nin, by_label=False)
        return self

    def SetLabel(self, newLabel, idx):
        """
        Set a new label for the bond at index :idx:

        Args:

            newLabel: The new label, it should be an integer.

            idx     : The index of the bond. when specified, the label of the bond at this index will be changed.

        Example:

            >>> g = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(4)],rowrank=1,labels=[5,6])
            >>> g.labels
            [5 6]


            Set "-1" to replace the original label "6" at index 1

            >>> g.SetLabel(-1,1)
            >>> g.labels
            [5 -1]

        """
        if not type(newLabel) is int or not type(idx) is int:
            raise TypeError("UniTensor.SetLabel", "newLabel and idx must be int.")

        if not idx < len(self.labels):
            raise ValueError("UniTensor.SetLabel", "idx exceed the number of bonds.")

        if newLabel in self.labels:
            raise ValueError("UniTensor.SetLabel", "newLabel [%d] already exists in the current UniTensor." % newLabel)

        self.labels[idx] = newLabel

    def SetLabels(self, newlabels):
        """
        Set new labels for all the bonds.

        Args:

            newLabels: The list of new labels, it should be a list or numpy array with size equal to the number of bonds of the UniTensor.

        Example:

            >>> g = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(4)],rowrank=1,labels=[5,6])
            >>> g.labels
            [5 6]

            Set new_label=[-1,-2] to replace the original label [5,6].

            >>> new_label=[-1,-2]
            >>> g.SetLabels(new_label)
            >>> g.labels
            [-1 -2]

        """
        if isinstance(newlabels, list):
            newlabels = np.array(newlabels)

        if not len(newlabels) == len(self.labels):
            raise ValueError("UniTensor.SetLabels",
                             "the length of newlabels does not match with the rank of UniTensor.")

        if len(np.unique(newlabels)) != len(newlabels):
            raise ValueError("UniTensor.SetLabels", "the newlabels contain duplicate entries.")

        self.labels = copy.copy(newlabels)

    def SetName(self, name):
        """
        Set the name of the UniTensor

        Args:

            name:
                a string.

        """
        if not isinstance(name, str):
            raise TypeError("UniTensor.str", "the name should be a string.")

        self.name = name

        return self

    def SetElem(self, elem):
        """
        Given 1D array of elements, set the elements stored in tensor as the same as the given ones. Note that elem can only be python-list or numpy

        Args:

            elem:
                The elements to be replace the content of the current UniTensor. It should be a 1D array.
                **Note** if the UniTensor is a tensor with symmetry, one should use UniTensor.PutBlock to set the elements.

        Example:
        ::
            Sz = tor10.UniTensor(bonds=[tor10.Bond(2),tor10.Bond(2)],rowrank=1,
                              dtype=torch.float64,
                              device=torch.device("cpu"))
            Sz.SetElem([1, 0,
                        0,-1 ])


        >>> print(Sz)
        Tensor name: 
        is_diag    : False
        tensor([[ 1.,  0.],
                [ 0., -1.]], dtype=torch.float64)

        """
        if not isinstance(elem, list) and not isinstance(elem, np.ndarray):
            raise TypeError("UniTensor.SetElem", "[ERROR]  elem can only be python-list or numpy")

        ## Qnum_ipoint [OKv03]
        if self.is_symm:
            raise Exception("UniTensor.SetElem", "[ERROR] the TN that has symm should use PutBlock.")

        if not len(elem) == self.Storage.numel():
            raise ValueError("UniTensor.SetElem", "[ERROR] number of elem is not equal to the # of elem in the tensor.")

        raw_elems = np.array(elem)
        if len(raw_elems.shape) != 1:
            raise Exception("UniTensor.SetElem", "[ERROR] can only accept 1D array of elements.")

        my_type = self.Storage.dtype
        my_shape = self.Storage.shape
        my_device = self.Storage.device
        self.Storage = torch.from_numpy(raw_elems).type(my_type).reshape(my_shape).to(my_device)

    def SetRowRank(self, new_rowrank):
        """
        Set the RowRank while keep the tensor indices.
            
            Args:
            
                new_rowrank: 
                    should be a unsigned int. 
                
                    [Note] for UniTensor with symmetry, it should have at least one bond in row-space and one bond in col-space. which means rowrank must >=1 and <= (rank of UniTensor)-1 

            Return:

                self

        """
        if self.is_symm:
            ##check:
            if new_rowrank < 1 or len(self.labels) - new_rowrank < 1:
                raise Exception("[ERROR]",
                                "SetRowRank for a tensor with symmetry must have at least 1 bond in row-space and 1 bond in col-space")
            self.rowrank = int(new_rowrank)
        else:
            if new_rowrank < 0 or new_rowrank > len(self.labels):
                raise Exception("[ERRROR]",
                                "Invalid Rowrank. Must >=0 and <= rank of tensor for non-symmetry UniTensor")
            self.rowrank = int(new_rowrank)

        self._check_braket()
        return self

    def Todense_(self):
        """
        Set the current UniTensor to dense matrix.
            [v0.3+] This only affect on UniTensor with non-symmetry with diag=True.

        Return:
            self

        Example:

            >>> a = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(3)],rowrank=1,is_diag=True)
            >>> a.SetElem([1,2,3])
            >>> print(a.is_diag)
            True

            >>> print(a)
            Tensor name: 
            is_diag    : True
            tensor([1., 2., 3.], dtype=torch.float64)

            >>> a.Todense_()
            >>> print(a.is_diag)
            False

            >>> print(a)
            Tensor name: 
            is_diag    : False
            tensor([[1., 0., 0.],
                    [0., 2., 0.],
                    [0., 0., 3.]], dtype=torch.float64)

        """
        if self.is_symm:
            raise Exception("UniTensor.Todense()", "[ERROR] cannot transform to dense for UniTensor with symmetry")

        if self.is_diag:
            self.Storage = torch.diag(self.Storage)
            self.is_diag = False

        return self

    def Todense(self):
        """
        Return a dense version of current UniTensor. This only affect on non-symmetric UniTensor with is_diag=True.
        
            [Note] for symmetric UniTensor, Todense cannot be called.

        Return:
            new UniTensor if current tensor is_diag=True, otherwise return self.

        Example:

            >>> a = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(3)],rowrank=1,is_diag=True)
            >>> print(a.is_diag)
            True

            >>> dense_a = a.Todense()
            >>> print(dense_a.is_diag)
            False

            >>> print(a.is_diag)
            True


        """
        if self.is_symm:
            raise Exception("UniTensor.Todense()", "[ERROR] cannot transform to dense form for symmetric UniTensor")

        if self.is_diag:
            out = copy.deepcopy(self)
            out.Todense_()
            return out
        else:
            return self

    def to_(self, device):
        """
        Set the current UniTensor to device

        Args:

            device:
                This should be an [torch.device]
                torch.device("cpu") for put the tensor on host (cpu)
                torch.device("cuda:x") for put the tensor on GPU with index x

        Return:
            
            self

        Example:

            Construct a tensor (default is on cpu)

            >>> a = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(4)],rowrank=1)

            Set to GPU.

            >>> a.to_(torch.device("cuda:0"))


        """
        if not isinstance(device, torch.device):
            raise TypeError("[ERROR] UniTensor.to()", "only support device argument in this version as torch.device")

        if self.device != device:
            if self.is_symm:
                for s in range(len(self.Storage)):
                    self.Storage[s] = self.Storage[s].to(device)
            else:
                self.Storage = self.Storage.to(device)

        return self

    def to(self, device):
        """
        Set the current UniTensor to device. If device is not the same with current tensor, return a new UniTensor,
        otherwise return self.

        Args:

            device:
                This should be an [torch.device]
                torch.device("cpu") for put the tensor on host (cpu)
                torch.device("cuda:x") for put the tensor on GPU with index x

        Return:
            
            Self if the device is the same as the current UniTensor. Otherwise, return a new UniTensor

        Example:

            Construct a tensor (default is on cpu)

            >>> a = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(4)],rowrank=1)

            Set to GPU.

            >>> b = a.to(torch.device("cuda:0"))
            >>> print(b is a)
            False

            >>> b = a.to(torch.device("cpu"))
            >>> print(b is a)
            True


        """
        if not isinstance(device, torch.device):
            raise TypeError("[ERROR] UniTensor.to()", "only support device argument in this version as torch.device")

        if self.device != device:
            out = copy.deepcopy(self)
            out.to_(device)
            return out
        else:
            return self

    def Print_diagram(self, bond_info=False):
        """
        This is the beauty print of the tensor diagram. Including the information of the current device
        ::
            1.The left hand side is always the in-bonds,representing the row-space when flatten as Matrix;
            the right hand side is always the Out-bonds, representing the column-space when flatten as Matrix.
            2.The number attached to the outside of each leg is the Bond-dimension.
            3.The number attached to the inside of each leg is the label.
            4.if all the bra-bonds are in row-space (in-bonds), and all ket-bonds are in col-space (out-bonds),
            the tensor is in "braket_form".
            5.if one permute bra-bonds that should be in-bonds to out-bonds, this will put the UniTensor in a
            "non-braket_form". the bond will have a "*" symbol on it.


        Args:
        
            bond_info [default: False]

                if set to True, the info of each bond will be printed.

        """
        print("-----------------------")
        print("tensor Name : %s" % self.name)
        print("tensor Rank : %d" % (len(self.labels)))
        print("has_symmetry: %s" % ("True" if self.is_symm else "False"))
        if self.is_symm:
            print("on device     : %s" % self.Storage[0].device)
        else:
            print("on device     : %s" % self.Storage.device)
            print("is_diag       : %s" % ("True" if self.is_diag else "False"))

        Nin = self.rowrank
        Nout = len(self.bonds) - self.rowrank
        if Nin > Nout:
            vl = Nin
        else:
            vl = Nout

        if self.braket is not None:
            print("braket_form : %s" % ("True" if self.is_braket else "False"))
            print("      |ket>               <bra| ")
            print("           ---------------      ")
            for i in range(vl):
                print("           |             |     ")
                if i < Nin:
                    if self.braket[i] == BondType[BD_KET]:
                        bks = "> "
                    else:
                        bks = "<*"
                    l = "%3d %s__" % (self.labels[i], bks)
                    llbl = "%-3d" % self.bonds[i].dim
                else:
                    l = "        "
                    llbl = "   "
                if i < Nout:
                    if self.braket[Nin + i] == BondType[BD_KET]:
                        bks = "*>"
                    else:
                        bks = " <"
                    r = "__%s %-3d" % (bks, self.labels[Nin + i])
                    rlbl = "%3d" % self.bonds[Nin + i].dim
                else:
                    r = "        "
                    rlbl = "   "
                print("   %s| %s     %s |%s" % (l, llbl, rlbl, r))
            print("           |             |     ")
            print("           ---------------     ")
        else:
            print("            -------------      ")
            for i in range(vl):
                if i == 0:
                    print("           /             \     ")
                else:
                    print("           |             |     ")
                if i < Nin:
                    bks = "__"
                    l = "%3d %s__" % (self.labels[i], bks)
                    llbl = "%-3d" % self.bonds[i].dim
                else:
                    l = "        "
                    llbl = "   "
                if i < Nout:
                    bks = "__"
                    r = "__%s %-3d" % (bks, self.labels[Nin + i])
                    rlbl = "%3d" % self.bonds[Nin + i].dim
                else:
                    r = "        "
                    rlbl = "   "
                print("   %s| %s     %s |%s" % (l, llbl, rlbl, r))
            print("           \             /     ")
            print("            -------------      ")

        if bond_info:
            for i in range(len(self.bonds)):
                print("lbl:%d " % (self.labels[i]), end="")
                print(self.bonds[i])

    def __str__(self):
        print("Tensor name: %s" % self.name)
        if self.braket is not None:
            print("braket_form : %s" % ("True" if self.is_braket else "False"))

        if self.is_symm:
            print("[Symmetry]")
            if self._contiguous:
                for b in range(len(self.Storage)):
                    print(self.Storage[b])
            else:

                out = self.Contiguous()
                for b in range(len(out.Storage)):
                    print(out.Storage[b])
                del out

                ## DEBUG >>>
                if DEBUG:
                    print("xxxxxxxxxxxxxxxxxxxxxx")
                    print("[DEBUG]")
                    print("Real memory:")
                    for b in range(len(self.Storage)):
                        print(self.Storage[b])
                    print("xxxxxxxxxxxxxxxxxxxxxx")
                ## <<<

        else:
            print("is_diag    : %s" % ("True" if self.is_diag else "False"))
            print(self.Storage)

        return ""

    def __repr__(self):
        print("Tensor name: %s" % self.name)
        if self.braket is not None:
            print("braket_form : %s" % ("True" if self.is_braket else "False"))

        if self.is_symm:
            print("[Symmetry]")
            if self._contiguous:
                for b in range(len(self.Storage)):
                    print(self.Storage[b])
            else:
                out = self.Contiguous()
                for b in range(len(out.Storage)):
                    print(out.Storage[b])
                del out

                ## DEBUG >>>
                if DEBUG:
                    print("xxxxxxxxxxxxxxxxxxxxxx")
                    print("Real memory:")
                    for b in range(len(self.Storage)):
                        print(self.Storage[b])
                    print("xxxxxxxxxxxxxxxxxxxxxx")
                ## <<<

        else:
            print("is_diag    : %s" % ("True" if self.is_diag else "False"))
            print(self.Storage)

        return ""

    def __len__(self):
        if self.is_symm:
            raise Exception("[ERROR]", "UniTensor with symmetry doesn't have property len")
        else:
            return len(self.Storage)

    def __eq__(self, rhs):
        """
            Compare two UniTensors.
            ::
                a == b

            where a & b are UniTensors.

            Note that this will only compare the shape of Storage. Not the content of torch tensor.


        """
        if isinstance(rhs, self.__class__):
            if self.is_symm != rhs.is_symm:
                return False

            if not (len(self.bonds) == len(rhs.bonds)):
                return False

            if not (all(self.bonds[i] == rhs.bonds[i] for i in range(len(self.bonds))) and all(
                    self.labels[i] == rhs.labels[i] for i in range(len(self.labels)))):
                return False

            if not self.rowrank == rhs.rowrank:
                return False

            if (self.braket is None) != (rhs.braket is None):
                return False

            if self.braket is None:
                if not (self.braket == rhs.braket).all():
                    return False

            if self.is_symm:
                iss = True
            else:
                iss = (self.is_diag == rhs.is_diag)
                iss = iss and (self.Storage.shape == rhs.Storage.shape)

            return iss

        else:
            raise ValueError("Bond.__eq__", "[ERROR] invalid comparison between Bond object and other type class.")

    def __ne__(self, other):
        return not (self == other)

    @property
    def device(self):
        """
            Return the device of UniTensor
            
            Return:

                torch.device
           
        """

        if self.is_symm:
            return self.Storage[0].device
        else:
            return self.Storage.device

    @property
    def dtype(self):
        """
            Return the device of UniTensor
            
            Return:
                torch.type 
        """
        if self.is_symm:
            return self.Storage[0].dtype
        else:
            return self.Storage.dtype

    @property
    def shape(self):
        """
            Return the shape of UniTensor

            Return:

                torch.Size
        """
        if self.is_symm:
            ## what to return ?
            # raise Exception("[DEvelope]")
            return torch.Size([self.bonds[z].dim for z in range(len(self.bonds))])
        else:
            if self.is_diag:
                return torch.Size([self.bonds[0].dim, self.bonds[0].dim])
            else:
                return self.Storage.shape

    ## Fill :
    def __getitem__(self, key):
        if self.is_symm:
            raise Exception("UniTensor.__getitem__",
                            "[ERROR] cannot use [] to getitem from a block-form tensor. Use get block first.")
        return From_torch(self.Storage[key], rowrank=0)

    def __setitem__(self, key, value):
        if self.is_symm:
            raise Exception("UniTensor.__setitem__",
                            "[ERROR] cannot use [] to setitem from a block-form tensor. Use get block first.")

        self.Storage[key] = value

    def item(self):
        """
        Get the python scalar from a UniTensor with one element

        Return:
            python scalar

        """
        if self.is_symm:
            raise TypeError("UniTensor.item", "[ERROR] cannot operate item() on symmetry tensor")
        else:
            if self.Storage.numel() != 1:
                raise TypeError("UniTensor.item",
                                "[ERROR] only one-element tensors can be converted to Python scalars.")

            return self.Storage.item()

    ## Math ::
    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self.is_symm != other.is_symm:
                raise TypeError("[ERROR]", "Cannot + two symm and non-symm UniTensor ")

            if self.is_symm:
                if self != other:
                    raise TypeError("[ERROR]", "Cannot + two symm tensors that have different symmetry structure.")
                if self.is_contiguous() and other.is_contiguous():
                    tmp = UniTensor(bonds=self.bonds,
                                    labels=self.labels,
                                    rowrank=self.rowrank,
                                    check=False)

                    tmp._mac(torch_tensor=[self.Storage[b] + other.Storage[b] for b in range(len(self.Storage))],
                             braket=self.braket,
                             sym_mappers=(self._mapper, self._inv_mapper,
                                          self._Ket_mapper_blks, self._Ket_invmapper_blks,
                                          self._Bra_mapper_blks, self._Bra_invmapper_blks,
                                          self._contiguous,
                                          self._accu_off_in,
                                          self._accu_off_out,
                                          self._block_qnums))


                else:
                    raise Exception("[ERROR]",
                                    "Two symmetry tensors can only add when both are contiguous.\n suggestion: Call .Contiguous() or .Contiguous_() before add")



            else:
                if not (self.is_braket is None) == (other.is_braket is None):
                    raise Exception("[ERROR]", "Cannot add non-braket-tag tensor with tagged tensor")

                if self.is_diag and other.is_diag:
                    tmp = UniTensor(bonds=self.bonds,
                                    labels=self.labels,
                                    rowrank=self.rowrank,
                                    check=False,
                                    is_diag=True)

                    tmp._mac(torch_tensor=self.Storage + other.Storage,
                             braket=self.braket)


                elif self.is_diag == False and other.is_diag == False:
                    tmp = UniTensor(bonds=self.bonds,
                                    labels=self.labels,
                                    rowrank=self.rowrank,
                                    check=False)
                    tmp._mac(torch_tensor=self.Storage + other.Storage,
                             braket=self.braket)

                else:
                    if self.is_diag:
                        tmp = UniTensor(bonds=self.bonds,
                                        labels=self.labels,
                                        rowrank=self.rowrank,
                                        check=False)
                        tmp._mac(torch_tensor=torch.diag(self.Storage) + other.Storage,
                                 braket=self.braket)
                    else:
                        tmp = UniTensor(bonds=self.bonds,
                                        labels=self.labels,
                                        rowrank=self.rowrank,
                                        check=False)
                        tmp._mac(torch_tensor=self.Storage + torch.diag(other.Storage),
                                 braket=self.braket)
        else:
            if self.is_symm:
                tmp = UniTensor(bonds=self.bonds,
                                labels=self.labels,
                                rowrank=self.rowrank,
                                check=False)

                tmp._mac(torch_tensor=[self.Storage[b] + other for b in range(len(self.Storage))],
                         braket=self.braket,
                         sym_mappers=(self._mapper, self._inv_mapper,
                                      self._Ket_mapper_blks, self._Ket_invmapper_blks,
                                      self._Bra_mapper_blks, self._Bra_invmapper_blks,
                                      self._contiguous,
                                      self._accu_off_in,
                                      self._accu_off_out,
                                      self._block_qnums))
            else:
                tmp = UniTensor(bonds=self.bonds,
                                labels=self.labels,
                                rowrank=self.rowrank,
                                check=False,
                                is_diag=self.is_diag)
                tmp._mac(torch_tensor=self.Storage + other,
                         braket=self.braket)
        return tmp

    def __radd__(self, other):
        ## U + U is handled by __add__, so we only need to process x + U here.
        if self.is_symm:
            tmp = UniTensor(bonds=self.bonds,
                            labels=self.labels,
                            rowrank=self.rowrank,
                            check=False)

            tmp._mac(braket=self.braket,
                     torch_tensor=[other + self.Storage[b] for b in range(len(self.Storage))],
                     sym_mappers=(self._mapper, self._inv_mapper,
                                  self._Ket_mapper_blks, self._Ket_invmapper_blks,
                                  self._Bra_mapper_blks, self._Bra_invmapper_blks,
                                  self._contiguous,
                                  self._accu_off_in,
                                  self._accu_off_out,
                                  self._block_qnums))

        else:
            tmp = UniTensor(bonds=self.bonds,
                            labels=self.labels,
                            rowrank=self.rowrank,
                            check=False,
                            is_diag=self.is_diag)
            tmp._mac(torch_tensor=other + self.Storage,
                     braket=self.braket)

        return tmp

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            if self.is_symm != other.is_symm:
                raise TypeError("[ERROR]", "[Cannot subtract symmetric and non-symmetric UniTensors]")

            if self.is_symm:
                if self != other:
                    raise TypeError("[ERROR]", "Cannot subtract symmetric tensors with different symmetry structure.")
                if self.is_contiguous() and other.is_contiguous():
                    tmp = UniTensor(bonds=self.bonds,
                                    labels=self.labels,
                                    rowrank=self.rowrank,
                                    check=False)

                    tmp._mac(braket=self.braket,
                             torch_tensor=[self.Storage[b] - other.Storage[b] for b in range(len(self.Storage))],
                             sym_mappers=(self._mapper, self._inv_mapper,
                                          self._Ket_mapper_blks, self._Ket_invmapper_blks,
                                          self._Bra_mapper_blks, self._Bra_invmapper_blks,
                                          self._contiguous,
                                          self._accu_off_in,
                                          self._accu_off_out,
                                          self._block_qnums))

                else:
                    raise Exception("[ERROR]",
                                    "Two symmetry tensors can only sub when both are contiguous.\n suggestion: Call .Contiguous() or .Contiguous_() before sub")

            else:
                if not (self.is_braket is None) == (other.is_braket is None):
                    raise Exception("[ERROR]", "Cannot sub non-braket-tag tensor with tagged tensor")

                if self.is_diag and other.is_diag:
                    tmp = UniTensor(bonds=self.bonds,
                                    labels=self.labels,
                                    rowrank=self.rowrank,
                                    check=False,
                                    is_diag=True)

                    tmp._mac(torch_tensor=self.Storage - other.Storage,
                             braket=self.braket)

                elif self.is_diag == False and other.is_diag == False:
                    tmp = UniTensor(bonds=self.bonds,
                                    labels=self.labels,
                                    rowrank=self.rowrank,
                                    check=False)
                    tmp._mac(torch_tensor=self.Storage - other.Storage,
                             braket=self.braket)
                else:
                    if self.is_diag:
                        tmp = UniTensor(bonds=self.bonds,
                                        labels=self.labels,
                                        rowrank=self.rowrank,
                                        check=False)
                        tmp._mac(torch_tensor=torch.diag(self.Storage) - other.Storage,
                                 braket=self.braket)
                    else:
                        tmp = UniTensor(bonds=self.bonds,
                                        labels=self.labels,
                                        rowrank=self.rowrank,
                                        check=False)
                        tmp._mac(braket=self.braket,
                                 torch_tensor=self.Storage - torch.diag(other.Storage))

        else:
            if self.is_symm:
                tmp = UniTensor(bonds=self.bonds,
                                labels=self.labels,
                                rowrank=self.rowrank,
                                check=False)
                tmp._mac(braket=self.braket,
                         torch_tensor=[self.Storage[b] - other for b in range(len(self.Storage))],
                         sym_mappers=(self._mapper, self._inv_mapper,
                                      self._Ket_mapper_blks, self._Ket_invmapper_blks,
                                      self._Bra_mapper_blks, self._Bra_invmapper_blks,
                                      self._contiguous,
                                      self._accu_off_in,
                                      self._accu_off_out,
                                      self._block_qnums))

            else:
                tmp = UniTensor(bonds=self.bonds,
                                labels=self.labels,
                                rowrank=self.rowrank,
                                check=False,
                                is_diag=self.is_diag)

                tmp._mac(torch_tensor=self.Storage - other,
                         braket=self.braket)

        return tmp

    def __rsub__(self, other):
        if self.is_symm:
            tmp = UniTensor(bonds=self.bonds,
                            labels=self.labels,
                            rowrank=self.rowrank,
                            check=False)

            tmp._mac(braket=self.braket,
                     torch_tensor=[other - self.Storage[b] for b in range(len(self.Storage))],
                     sym_mappers=(self._mapper, self._inv_mapper,
                                  self._Ket_mapper_blks, self._Ket_invmapper_blks,
                                  self._Bra_mapper_blks, self._Bra_invmapper_blks,
                                  self._contiguous,
                                  self._accu_off_in,
                                  self._accu_off_out,
                                  self._block_qnums))
        else:
            tmp = UniTensor(bonds=self.bonds,
                            labels=self.labels,
                            rowrank=self.rowrank,
                            check=False,
                            is_diag=self.is_diag)
            tmp._mac(braket=self.braket,
                     torch_tensor=other - self.Storage)
        return tmp

    ""

    def Whole_transpose(self):
        """
            If the UniTensor is tagged, exchange the bra/ket tags on each bond, and transpose (rowspace and colspace) by referencing to the "rowrank".
            
            Return:
                UniTensor, shared the same type with each bond's tag bra <-> ket exchanged.

        """
        out = copy.deepcopy(self)
        if self.is_symm:
            ## symmetry Tensor:
            for b in range(len(self.bonds)):
                if out.bonds[b].bondType == BD_KET:
                    out.bonds[b].bondType = BD_BRA
                else:
                    out.bonds[b].bondType = BD_KET
            out.braket *= -1
            tmp = np.roll(np.arange(len(out.bonds)).astype(np.int), -out.rowrank)
            out.Permute(tmp, rowrank=len(out.bonds) - out.rowrank)

        else:
            if self.braket is None:
                ## untagged tensor
                tmp = np.roll(np.arange(len(out.bonds)).astype(np.int), -out.rowrank)
                out.Permute(tmp, rowrank=len(out.bonds) - out.rowrank)
                return out
            else:
                ## tagged nonsymm Tensor:                
                for b in range(len(self.bonds)):
                    if out.bonds[b].bondType == BD_KET:
                        out.bonds[b].bondType = BD_BRA
                    else:
                        out.bonds[b].bondType = BD_KET
                out.braket *= -1
                tmp = np.roll(np.arange(len(out.bonds)).astype(np.int), -out.rowrank)
                out.Permute(tmp, rowrank=len(out.bonds) - out.rowrank)

        return out

    ""

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            if self.is_symm != other.is_symm:
                raise TypeError("[ERROR]", "Cannot * two symm and non-symm UniTensor")
            if self.is_symm:
                if self != other:
                    raise TypeError("[ERROR]", "Cannot * two symm tensors that have different symmetry structure.")
                if self.is_contiguous() and other.is_contiguous():
                    tmp = UniTensor(bonds=self.bonds,
                                    labels=self.labels,
                                    rowrank=self.rowrank,
                                    check=False)

                    tmp._mac(braket=self.braket,
                             torch_tensor=[self.Storage[b] * other.Storage[b] for b in range(len(self.Storage))],
                             sym_mappers=(self._mapper, self._inv_mapper,
                                          self._Ket_mapper_blks, self._Ket_invmapper_blks,
                                          self._Bra_mapper_blks, self._Bra_invmapper_blks,
                                          self._contiguous,
                                          self._accu_off_in,
                                          self._accu_off_out,
                                          self._block_qnums))
                else:
                    raise Exception("[ERROR]",
                                    "Two symmetry tensors can only mul when both are contiguous.\n suggestion: Call .Contiguous() or .Contiguous_() before mul")
            else:
                if not (self.is_braket is None) == (other.is_braket is None):
                    raise Exception("[ERROR]", "Cannot mul non-braket-tag tensor with tagged tensor")

                if self.is_diag and other.is_diag:
                    tmp = UniTensor(bonds=self.bonds,
                                    labels=self.labels,
                                    rowrank=self.rowrank,
                                    check=False,
                                    is_diag=True)
                    tmp._mac(torch_tensor=self.Storage * other.Storage,
                             braket=self.braket)
                elif self.is_diag == False and other.is_diag == False:
                    tmp = UniTensor(bonds=self.bonds,
                                    labels=self.labels,
                                    rowrank=self.rowrank,
                                    check=False)
                    tmp._mac(torch_tensor=self.Storage * other.Storage,
                             braket=self.braket)
                else:
                    if self.is_diag:
                        tmp = UniTensor(bonds=self.bonds,
                                        labels=self.labels,
                                        rowrank=self.rowrank,
                                        check=False)
                        tmp._mac(torch_tensor=torch.diag(self.Storage) * other.Storage,
                                 braket=self.braket)
                    else:
                        tmp = UniTensor(bonds=self.bonds,
                                        labels=self.labels,
                                        rowrank=self.rowrank,
                                        check=False)
                        tmp._mac(torch_tensor=self.Storage * torch.diag(other.Storage),
                                 braket=self.braket)
        else:
            if self.is_symm:
                tmp = UniTensor(bonds=self.bonds,
                                labels=self.labels,
                                rowrank=self.rowrank,
                                check=False)

                tmp._mac(braket=self.braket,
                         torch_tensor=[self.Storage[b] * other for b in range(len(self.Storage))],
                         sym_mappers=(self._mapper, self._inv_mapper,
                                      self._Ket_mapper_blks, self._Ket_invmapper_blks,
                                      self._Bra_mapper_blks, self._Bra_invmapper_blks,
                                      self._contiguous,
                                      self._accu_off_in,
                                      self._accu_off_out,
                                      self._block_qnums))


            else:
                tmp = UniTensor(bonds=self.bonds,
                                labels=self.labels,
                                rowrank=self.rowrank,
                                check=False,
                                is_diag=self.is_diag)
                tmp._mac(braket=self.braket,
                         torch_tensor=self.Storage * other)
        return tmp

    def __rmul__(self, other):
        if self.is_symm:
            tmp = UniTensor(bonds=self.bonds,
                            labels=self.labels,
                            rowrank=self.rowrank,
                            check=False)

            tmp._mac(braket=self.braket,
                     torch_tensor=[other * self.Storage[b] for b in range(len(self.Storage))],
                     sym_mappers=(self._mapper, self._inv_mapper,
                                  self._Ket_mapper_blks, self._Ket_invmapper_blks,
                                  self._Bra_mapper_blks, self._Bra_invmapper_blks,
                                  self._contiguous,
                                  self._accu_off_in,
                                  self._accu_off_out,
                                  self._block_qnums))
        else:
            tmp = UniTensor(bonds=self.bonds,
                            labels=self.labels,
                            rowrank=self.rowrank,
                            check=False,
                            is_diag=self.is_diag)
            tmp._mac(torch_tensor=other * self.Storage,
                     braket=self.braket)
        return tmp

    def __pow__(self, other):
        if self.is_symm:
            # raise Exception("[Develope][check impl]")
            tmp = UniTensor(bonds=self.bonds,
                            labels=self.labels,
                            rowrank=self.rowrank,
                            check=False)

            tmp._mac(braket=self.braket,
                     torch_tensor=[self.Storage[b] ** other for b in range(len(self.Storage))],
                     sym_mappers=(self._mapper, self._inv_mapper,
                                  self._Ket_mapper_blks, self._Ket_invmapper_blks,
                                  self._Bra_mapper_blks, self._Bra_invmapper_blks,
                                  self._contiguous, self._accu_off_in, self._accu_off_out, self._block_qnums))
            return tmp
        else:
            tmp = UniTensor(bonds=self.bonds,
                            labels=self.labels,
                            rowrank=self.rowrank,
                            check=False,
                            is_diag=self.is_diag)

            tmp._mac(braket=self.braket, torch_tensor=self.Storage ** other)
            return tmp

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            if self.is_symm != other.is_symm:
                raise TypeError("[ERROR]", "Cannot / two symm and non-symm UniTensor.")

            if self.is_symm:
                if self != other:
                    raise TypeError("[ERROR]", "Cannot / two symm tensors that have different symmetry structure.")
                if self.is_contiguous() and other.is_contiguous():
                    tmp = UniTensor(bonds=self.bonds,
                                    labels=self.labels,
                                    rowrank=self.rowrank,
                                    check=False)

                    tmp._mac(braket=self.braket,
                             torch_tensor=[self.Storage[b] / other.Storage[b] for b in range(len(self.Storage))],
                             sym_mappers=(self._mapper, self._inv_mapper,
                                          self._Ket_mapper_blks, self._Ket_invmapper_blks,
                                          self._Bra_mapper_blks, self._Bra_invmapper_blks,
                                          self._contiguous,
                                          self._accu_off_in,
                                          self._accu_off_out,
                                          self._block_qnums))
                else:
                    raise Exception("[ERROR]",
                                    "Two symmetry tensors can only mul when both are contiguous.\n suggestion: Call .Contiguous() or .Contiguous_() before mul")
            else:
                if not (self.is_braket is None) == (other.is_braket is None):
                    raise Exception("[ERROR]", "Cannot / non-braket-tag tensor with tagged tensor")

                if self.is_diag:
                    if other.is_diag:
                        tmp = UniTensor(bonds=self.bonds,
                                        labels=self.labels,
                                        rowrank=self.rowrank,
                                        check=False,
                                        is_diag=True)

                        tmp._mac(braket=self.braket,
                                 torch_tensor=self.Storage / other.Storage)
                    else:
                        tmp = UniTensor(bonds=self.bonds,
                                        labels=self.labels,
                                        rowrank=self.rowrank,
                                        check=False)

                        tmp._mac(torch_tensor=torch.diag(self.Storage) / other.Storage,
                                 braket=self.braket)
                else:
                    if other.is_diag:
                        tmp = UniTensor(bonds=self.bonds,
                                        labels=self.labels,
                                        rowrank=self.rowrank,
                                        check=False)

                        tmp._mac(torch_tensor=self.Storage / torch.diag(other.Storage),
                                 braket=self.braket)
                    else:
                        tmp = UniTensor(bonds=self.bonds,
                                        labels=self.labels,
                                        rowrank=self.rowrank,
                                        check=False)

                        tmp._mac(braket=self.braket,
                                 torch_tensor=self.Storage / other.Storage)

        else:
            if self.is_symm:
                tmp = UniTensor(bonds=self.bonds,
                                labels=self.labels,
                                rowrank=self.rowrank,
                                check=False)

                tmp._mac(braket=self.braket,
                         torch_tensor=[self.Storage[b] / other for b in range(len(self.Storage))],
                         sym_mappers=(self._mapper, self._inv_mapper,
                                      self._Ket_mapper_blks, self._Ket_invmapper_blks,
                                      self._Bra_mapper_blks, self._Bra_invmapper_blks,
                                      self._contiguous,
                                      self._accu_off_in,
                                      self._accu_off_out,
                                      self._block_qnums))
            else:
                tmp = UniTensor(bonds=self.bonds,
                                labels=self.labels,
                                rowrank=self.rowrank,
                                check=False,
                                is_diag=self.is_diag)
                tmp._mac(braket=self.braket,
                         torch_tensor=self.Storage / other)

        return tmp

    ## This is the same function that behaves as the memberfunction.
    def Svd(self):
        """
            This is the member function of Svd, see tor10.linalg.Svd()
        """
        if self.is_symm:
            raise Exception("UniTensor.Svd",
                            "[ERROR] cannot perform Svd on a symmetry,block-form tensor. use GetBlock() first and perform svd on the Block.")

        if self.braket is not None:
            raise Exception("UniTensor.Svd", "[ERROR] cannot perform Svd on a bra-ket tagged tensor.")

        return linalg.Svd(self)

    def Svd_truncate(self,keepdim=None):
        """
            This is the member function of Svd_truncate, see tor10.linalg.Svd_truncate()
        """
        if self.is_symm:
            raise Exception("UniTensor.Svd_truncate",
                            "[ERROR] cannot perform Svd on a symmetry,block-form tensor. use GetBlock() first and perform svd on the Block.")

        return linalg.Svd_truncate(self,keepdim)

    def Norm(self):
        """
            This is the member function of Norm, see tor10.linalg.Norm
        """
        if self.is_symm:
            raise Exception("UniTensor.Norm",
                            "[ERROR] cannot perform Norm on a symmetry,block-form tensor. use GetBlock() first and perform svd on the Block.")

        return linalg.Norm(self)

    def Det(self):
        """
            This is the member function of Det, see tor10.linalg.Det
        """
        if self.is_symm:
            raise Exception("UniTensor.Det",
                            "[ERROR] cannot perform Det on a symmetry, block-form tensor. use GetBlock() first and perform det on the Block.")

        return linalg.Det(self)

    def Matmul(self, b):
        """
            This is the member function of Matmul, see tor10.linalg.Matmul
        """
        if self.is_symm:
            raise Exception("UniTensor.Matmul",
                            "[ERROR] cannot perform MatMul on a symmetry, block-form tensor. use GetBlock() first and perform matmul on the Block.")

        return linalg.Matmul(self, b)

    ## Extended Assignment:
    def __iadd__(self, other):
        if isinstance(other, self.__class__):
            if self.is_symm != other.is_symm:
                raise TypeError("[ERROR]", "cannot += symm and non-symm UniTensors")

            if self.is_symm:

                if self != other:
                    raise TypeError("[ERROR]", "Cannot + two symm tensors that have different symmetry structure.")
                if self.is_contiguous() and other.is_contiguous():
                    for b in range(len(self.Storage)):
                        self.Storage[b] += other.Storage[b]

                else:
                    raise Exception("[ERROR]",
                                    "Two symmetry tensors can only add when both are contiguous.\n suggestion: Call .Contiguous() or .Contiguous_() before add")

            else:
                if (self.braket is None) != (other.braket is None):
                    raise Exception("[ERROR]", "cannot += non-braket-tag tensor with tagged tensor")

                if self.is_diag == other.is_diag:
                    self.Storage += other.Storage
                else:
                    if self.is_diag:
                        self.Storage = torch.diag(self.Storage) + other.Storage
                        self.is_diag = False
                    else:
                        self.Storage += torch.diag(other.Storage)

        else:
            if self.is_symm:
                for b in range(len(self.Storage)):
                    self.Storage[b] += other
            else:
                self.Storage += other

        return self

    def __isub__(self, other):
        if isinstance(other, self.__class__):
            if self.is_symm != other.is_symm:
                raise TypeError("[ERROR]", "cannot -= symm and non-symm UniTensors")

            if self.is_symm:
                if self != other:
                    raise TypeError("[ERROR]", "Cannot - two symm tensors that have different symmetry structure.")
                if self.is_contiguous() and other.is_contiguous():
                    for b in range(len(self.Storage)):
                        self.Storage[b] -= other.Storage[b]

                else:
                    raise Exception("[ERROR]",
                                    "Two symmetry tensors can only sub when both are contiguous.\n suggestion: Call .Contiguous() or .Contiguous_() before sub")
            else:
                if (self.braket is None) != (other.braket is None):
                    raise Exception("[ERROR]", "cannot -= non-braket-tag tensor with tagged tensor")

                if self.is_diag == other.is_diag:
                    self.Storage -= other.Storage
                else:
                    if self.is_diag:
                        self.Storage = torch.diag(self.Storage) + other.Storage
                        self.is_diag = False
                    else:
                        self.Storage -= torch.diag(other.Storage)

        else:
            if self.is_symm:
                for b in range(len(self.Storage)):
                    self.Storage[b] -= other
            else:
                self.Storage -= other

        return self

    def __imul__(self, other):
        if isinstance(other, self.__class__):
            if self.is_symm != other.is_symm:
                raise TypeError("[ERROR]", "cannot *= symm and non-symm UniTensors")

            if self.is_symm:
                if self != other:
                    raise TypeError("[ERROR]", "Cannot * two symm tensors that have different symmetry structure.")
                if self.is_contiguous() and other.is_contiguous():
                    for b in range(len(self.Storage)):
                        self.Storage[b] *= other.Storage[b]

                else:
                    raise Exception("[ERROR]",
                                    "Two symmetry tensors can only mul when both are contiguous.\n suggestion: Call .Contiguous() or .Contiguous_() before mul")

            else:
                if (self.braket is None) != (other.braket is None):
                    raise Exception("[ERROR]", "cannot -= non-braket-tag tensor with tagged tensor")

                if self.is_diag == other.is_diag:
                    self.Storage *= other.Storage
                else:
                    if self.is_diag:
                        self.Storage = torch.diag(self.Storage) * other.Storage
                        self.is_diag = False
                    else:
                        self.Storage *= torch.diag(other.Storage)
        else:
            if self.is_symm:
                for b in range(len(self.Storage)):
                    self.Storage[b] *= other
            else:
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
        # v0.3+ OK.
        _Randomize(self)

        return self

    def CombineBonds(self, X_to_combine, new_label=None, permute_back=False, by_label=True):
        """
        This function combines the bonds in input UniTensor [a] by the specified labels [label].

        [Note][v0.3+] that ket-bonds can only be combine with ket-bonds, bra-bonds can only combine with bra-bonds.

        Args:

            labels_to_combine:
                labels that to be combined. It should be a int list / numpy array of the label. All the bonds with specified labels in the current UniTensor  will be combined

            new_label [default=None]
                This should be an integer, for floating point number, it will be truncated to integer.

                if new_label is set to None, the combined bond will have label as the bond in the to-be-combined bonds that has the smallest LABEL in input tensor.

                if new_label is set, the combined bond will have label [new_label]
        
            permuted_back[False]:
                this state if the combine bond should be permuted back or not. If false, the combined bond will always be presented as the first bond.


        Example:

            1. Combine Bond for an non-symmetric tensor.

            >>> bds_x = [tor10.Bond(5),tor10.Bond(5),tor10.Bond(3)]
            >>> x = tor10.UniTensor(bonds=bds_x, rowrank=2, labels=[4,3,5])
            >>> y = tor10.UniTensor(bonds=bds_x, rowrank=2, labels=[4,3,5])
            >>> x.Print_diagram()
            tensor Name : 
            tensor Rank : 3
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 4 ____| 5         3 |____ 5  
                       |             |     
                 3 ____| 5           |        
                       \             /     
                        -------------      
            lbl:4 Dim = 5 |
            REG     :
            _
            lbl:3 Dim = 5 |
            REG     :
            _
            lbl:5 Dim = 3 |
            REG     :


            * combine bond with label "3" into "5"
            
            >>> x.CombineBonds([5,3])
            >>> x.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 2
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 4 ____| 5        15 |____ 5  
                       \             /     
                        -------------      
            lbl:4 Dim = 5 |
            REG     :
            _
            lbl:5 Dim = 15 |
            REG     :


            * combine bond with label "5" into "3"

            >>> y.CombineBonds([3,5])
            >>> y.Print_diagram()
            tensor Name : 
            tensor Rank : 2
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 4 ____| 5           |        
                       |             |     
                 3 ____| 15          |        
                       \             /     
                        -------------      
            lbl:4 Dim = 5 |
            REG     :
            _
            lbl:3 Dim = 15 |
            REG     :

            
            >>> z  = tor10.UniTensor(bonds=bds_x*2, rowrank=3, labels=[4,3,5,6,7,8])
            >>> z2 = tor10.UniTensor(bonds=bds_x*2, rowrank=3, labels=[4,3,5,6,7,8])
            >>> z.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 6
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 4 ____| 5         5 |____ 6  
                       |             |     
                 3 ____| 5         5 |____ 7  
                       |             |     
                 5 ____| 3         3 |____ 8  
                       \             /     
                        ------------- 
            
            >>> z.CombineBonds([4,5,6])
            >>> z.Print_diagram()
            tensor Name : 
            tensor Rank : 4
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 4 ____| 225       5 |____ 3  
                       |             |     
                       |           5 |____ 7  
                       |             |     
                       |           3 |____ 8  
                       \             /     
                        -------------   

            >>> z2.CombineBonds([4,5,6],permute_back=True)
            >>> z2.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 4
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 4 ____| 225       5 |____ 7  
                       |             |     
                 3 ____| 5         3 |____ 8  
                       \             /     
                        -------------
        """
        if len(X_to_combine) < 2:
            raise ValueError("CombineBonds", "[ERROR] the number of bonds to combine should be greater than one.")

        # checking :
        if by_label:
            same_lbls, x_ind, _ = np.intersect1d(self.labels, X_to_combine, return_indices=True)

            if len(same_lbls) != len(X_to_combine):
                raise Exception("[ERROR] not all the label appears in the current tensor.")

            idxs_to_combine = []
            for l in X_to_combine:
                idxs_to_combine.append(np.argwhere(self.labels == l).flatten()[0])

            idxs_to_combine = np.array(idxs_to_combine, dtype=np.int)
            # print(idxs_to_combine)
        else:
            if not all(X_to_combine < len(a.labels)):
                raise Exception("[ERROR] index out of bound")

            idxs_to_combine = np.array(X_to_combine, dtype=np.int)
        # print(idxs_to_combine)

        _CombineBonds(self, idxs_to_combine, new_label, permute_back)

    def Contiguous_(self):
        """
        Make the memory contiguous. This is similar as pytorch's contiguous_().
        Because of  Permute does not change the memory layout, after permute, only the shape of UniTensor is changed,
        the underlying memory layout does not change.
        This UniTensor under this condition is called "non-contiguous".
        When call the Contiguous_(), the memory will be moved to match the shape of UniTensor.
        *Note* Normally, it is not necessary to call contiguous. Most of the linalg function implicitly make the
        UniTensor contiguous. If one calls a function that requires a contiguous tensor,
        the error will be raised and you know you have to put UniTensor.Contiguous() or UniTensor.Contiguous_() there.

        Return:
            self

        Example:

            >>> bds_x = [tor10.Bond(5),tor10.Bond(5),tor10.Bond(3)]
            >>> x = Tt.UniTensor(bonds=bds_x,rowrank=1, labels=[4,3,5])
            >>> print(x.is_contiguous())
            True

            >>> x.Permute([0,2,1],rowrank=1)
            >>> print(x.is_contiguous())
            False

            >>> x.Contiguous_()
            >>> print(x.is_contiguous())
            True

        """
        if self.is_symm:
            # raise Exception("[Develope]")
            if self._contiguous:
                return self
            else:
                out = self.Contiguous()
                out.name = self.name
                self.__dict__.update(out.__dict__)
                return self

        else:
            self.Storage = self.Storage.contiguous()

        return self

    def Contiguous(self):
        """
        Make the memory contiguous. This is similar as pytorch's contiguous().
        Because of the Permute does not move the memory, after permute, only the shape of UniTensor is changed, the underlying memory does not change. The UniTensor in this status is called "non-contiguous" tensor.
        When call the Contiguous(), the memory will be moved to match the shape of UniTensor.
        
        if the current tensor is already in contiguous, return self. Otherwise, return a new tensor.


        Return:
            self

        Example:

            >>> bds_x = [tor10.Bond(5),tor10.Bond(5),tor10.Bond(3)]
            >>> x = Tt.UniTensor(bonds=bds_x,rowrank=1, labels=[4,3,5])
            >>> print(x.is_contiguous())
            True

            >>> x.Permute([0,2,1],rowrank=1)
            >>> print(x.is_contiguous())
            False

            >>> y = x.Contiguous()
            >>> print(y.is_contiguous())
            True

            >>> print(x.is_contiguous())
            False

        """
        if self.is_symm:
            # raise Exception("[Develope]")
            if self._contiguous:
                return self
            else:
                out = UniTensor(bonds=self.bonds,
                                labels=self.labels,
                                rowrank=self.rowrank,
                                device=self.device,
                                dtype=self.dtype)

                out._mac(braket=self.braket)

                out_bd_dims = np.array([out.bonds[x].dim for x in range(out.rowrank)], dtype=np.int)

                ## copy elemenets:  
                for b in range(len(self.Storage)):
                    oldshape = self.Storage[b].shape
                    for i in range(oldshape[0]):
                        for j in range(oldshape[1]):
                            oldidx = np.concatenate((self._Ket_invmapper_blks[b][i], self._Bra_invmapper_blks[b][j]))
                            newidx = oldidx[self._mapper]
                            #
                            new_row = int(np.sum(out._accu_off_in * newidx[:out.rowrank]))
                            new_col = int(np.sum(out._accu_off_out * newidx[out.rowrank:]))
                            b_id_in = out._Ket_mapper_blks[new_row]
                            b_id_out = out._Bra_mapper_blks[new_col]

                            ## [DEBUG] >>>>
                            if DEBUG:
                                if b_id_in[0] < 0 or b_id_out[0] < 0:
                                    raise Exception("[ERROR][DEBUG][Internal check neg pos]")
                                if b_id_in[0] != b_id_out[0]:
                                    print(b_id_in[0], b_id_out[0])
                                    print("[ERROR!][DEBUG][Internal check un-matched block]")
                                    exit(1)
                            ## <<<<
                            out.Storage[b_id_in[0]][b_id_in[1], b_id_out[1]] = self.Storage[b][i, j]
                # out._contiguous = True
                return out

        else:
            if self.is_contiguous():
                return self
            else:
                tmp = UniTensor(bonds=self.bonds,
                                labels=self.labels,
                                is_diag=self.is_diag,
                                rowrank=self.rowrank,
                                check=False)

                tmp._mac(braket=self.braket,
                         torch_tensor=self.Storage.contiguous())
                return tmp

    def is_contiguous(self):
        """
        Return the status of memory contiguous.

        Return:
            bool, if True, then the Storage of UniTensor is contiguous. if False, then the Storage of UiTensor is non-contiguous.

        """
        if self.is_symm:
            return self._contiguous
        else:
            return self.Storage.is_contiguous()

    def Permute(self, mapper, rowrank=None, by_label=False):
        """
        Permute the bonds of the UniTensor.
            
            [Note] the computation complexity of Permute is O(1) which is very fast. The permute will not change the underlying memory layout. It will put the tensor into a "non-contiguous" status. Call Contiguous() or Contiguous_() when actually need to move memory.


        Args:
            mapper:
                a python list or 1d numpy array with integer type elements that the UniTensor permute accordingly.
                If by_label=False, the in_mapper will use index as mapper.

            by_label: [default False]
                bool, when True, the mapper using the labels. When False, the mapper using the index.

            rowrank: [default: current rowrank]
                uint, the rank of row space. If not set, it is equal to the current Tensor's rank of row space.

        Return:

            self

        Example:

            >>> bds_x = [tor10.Bond(6),tor10.Bond(5),tor10.Bond(4),tor10.Bond(3),tor10.Bond(2)]
            >>> x = tor10.UniTensor(bonds=bds_x, rowrank=3,labels=[1,3,5,7,8])
            >>> y = copy.deepcopy(x)
            >>> z = copy.deepcopy(x)
            >>> x.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 5
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 1 ____| 6         3 |____ 7  
                       |             |     
                 3 ____| 5         2 |____ 8  
                       |             |     
                 5 ____| 4           |        
                       \             /     
                        ------------- 

            >>> x.Permute([0,2,1,4,3])
            >>> x.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 5
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 1 ____| 6         2 |____ 8  
                       |             |     
                 5 ____| 4         3 |____ 7  
                       |             |     
                 3 ____| 5           |        
                       \             /     
                        -------------
 
            >>> y.Permute([3,1,5,7,8],by_label=True)
            >>> y.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 5
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 3 ____| 5         3 |____ 7  
                       |             |     
                 1 ____| 6         2 |____ 8  
                       |             |     
                 5 ____| 4           |        
                       \             /     
                        -------------

            >>> z.Permute([3,1,5,7,8],rowrank=2,by_label=True)
            >>> z.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 5
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 3 ____| 5         4 |____ 5  
                       |             |     
                 1 ____| 6         3 |____ 7  
                       |             |     
                       |           2 |____ 8  
                       \             /     
                        -------------


 
        """
        ## check
        if not (isinstance(mapper, list) or isinstance(mapper, np.ndarray)):
            raise TypeError("UniTensor.Permute", "[ERROR] mapper should be an 1d python list or numpy array.")
        if len(mapper) != len(self.bonds):
            raise ValueError("UniTensor.Permute", "[ERROR] len(mapper) should equal to Tensor rank")

        ## check duplicate:
        if len(mapper) != len(np.unique(mapper)):
            raise ValueError("UniTensor.Permute", "[ERROR] mapper contain duplicate elements.")

        if by_label:
            DD = dict(zip(self.labels, np.arange(len(self.labels))))

            if not all(lbl in self.labels for lbl in mapper):
                raise Exception("UniTensor.Permute",
                                "[ERROR] by_label=True but mapper contain invalid labels not appear in the UniTensor label")
            idx_mapper = np.array([DD[x] for x in mapper])
        else:
            idx_mapper = np.array(mapper).astype(np.int)

        self.labels = self.labels[idx_mapper]
        self.bonds = self.bonds[idx_mapper]
        if self.braket is not None:
            self.braket = self.braket[idx_mapper]

        if rowrank is not None:
            if rowrank < 0:
                raise ValueError("UniTensor.Permute", "rowrank must >=0")

            self.rowrank = rowrank

        ## check braket_form:
        self._check_braket()

        ## master switch
        if self.is_symm:
            # raise Exception("[Developing]")
            self._mapper = self._mapper[idx_mapper]
            Arr_range = np.arange(len(self._mapper)).astype(np.int)
            if (self._mapper == Arr_range).all():
                self._contiguous = True
            else:
                self._contiguous = False

            self._inv_mapper = np.zeros(len(self._mapper)).astype(np.int)
            self._inv_mapper[self._mapper] = Arr_range
            self._inv_mapper = self._inv_mapper.astype(np.int)

            b_tqin, b_tqout = self.GetTotalQnums(physical=False)
            tqin_uni = b_tqin.GetUniqueQnums()
            tqout_uni = b_tqout.GetUniqueQnums()
            self._block_qnums = _fx_GetCommRows(tqin_uni, tqout_uni)

        else:

            if self.is_diag:
                if self.rowrank != 1:
                    raise Exception("UniTensor.Permute",
                                    "[ERROR] UniTensor.is_diag=True must have rowrank==1\n" + "Suggest, call Todense()")

            else:
                # print(idx_mapper)
                # print(self.Storage)
                self.Storage = self.Storage.permute(tuple(idx_mapper))

        return self

    def Reshape(self, dimer, rowrank, new_labels=None):
        """
        Return a new reshaped UniTensor into the shape specified as [dimer], with the first [rowrank] Bonds as bra-bond and other bonds as ket-bond.

        [Note] 

            1.Reshaping a UniTensor physically re-define the new basis, which construct a new physical definition tensor that has the same element.

            2.Reshape can only operate on an untagged tensor.

        Args:

            dimer:
                The new shape of the UniTensor. This should be a python list.

            rowrank:
                The number of bonds in row space.

            new_labels:
                The new labels that will be set for new bonds after reshape.

        reture:

            UniTensor

        Example:

            >>> bds_x = [tor10.Bond(6),tor10.Bond(5),tor10.Bond(3)]
            >>> x = tor10.UniTensor(bonds=bds_x, rowrank=1,labels=[4,3,5])
            >>> x.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 3
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 4 ____| 6         5 |____ 3  
                       |             |     
                       |           3 |____ 5  
                       \             /     
                        ------------- 

            >>> y = x.Reshape([2,3,5,3],new_labels=[1,2,3,-1],rowrank=2)
            >>> y.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 4
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 1 ____| 2         5 |____ 3  
                       |             |     
                 2 ____| 3         3 |____ -1 
                       \             /     
                        -------------  


        """
        if self.is_symm:
            raise TypeError("UniTensor.Reshape", "[ERROR] Cannot perform Reshape on a symmetry Tensor")

        if self.is_diag:
            raise Exception("UniTensor.Reshape", "[ERROR] UniTensor.is_diag=True cannot be Reshape.\n" +
                            "[Suggest] Call UniTensor.Todense()")

        if self.braket is not None:
            raise Exception("UniTensor.Reshape",
                            "[ERROR] UniTensor.Reshape can only operate on a [untagged] tensor with regular bonds (BD_REG).")

        if not isinstance(dimer, list):
            raise TypeError("UniTensor.Reshape", "[ERROR] mapper should be an python list.")

        new_Storage = copy.deepcopy(self.Storage)

        new_Storage = new_Storage.reshape(dimer)
        if new_labels is None:
            new_labels = np.arange(len(dimer))

        tmp = UniTensor(bonds=np.array([Bond(dimer[i]) for i in range(len(dimer))]),
                        labels=new_labels,
                        rowrank=rowrank,
                        check=False)

        tmp._mac(torch_tensor=new_Storage)

        return tmp

    def Reshape_(self, dimer, rowrank, new_labels=None):
        """
        Inplace version of Reshape. 
        Reshape UniTensor into the shape specified as [dimer], with the first [rowrank] Bonds as bra-bond and other bonds as ket-bond.

        [Note] 

            1.Reshapeing a UniTensor physically re-define the bra-ket basis space, which construct a new physical definition tensor that has the same element.

            2.Reshape can only operate on an untagged tensor.

        Args:

            dimer:
                The new shape of the UniTensor. This should be a python list.

            rowrank:
                The number of bonds in row space.

            new_labels [option]:
                The new labels that will be set for new bonds after reshape. If not set, the label will be initialize using default enumerate rule.

        Return:

            self
    
        Example:

            >>> bds_x = [tor10.Bond(6),tor10.Bond(5),tor10.Bond(3)]
            >>> x = tor10.UniTensor(bonds=bds_x, rowrank=1,labels=[4,3,5])
            >>> x.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 3
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 4 ____| 6         5 |____ 3  
                       |             |     
                       |           3 |____ 5  
                       \             /     
                        ------------- 

            >>> x.Reshape_([2,3,5,3],new_labels=[1,2,3,-1],rowrank=2)
            >>> x.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 4
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 1 ____| 2         5 |____ 3  
                       |             |     
                 2 ____| 3         3 |____ -1 
                       \             /     
                        -------------


        """
        if self.is_symm:
            raise TypeError("UniTensor.Reshape", "[ERROR] Cannot perform Reshape on a symmetry Tensor")

        if self.is_diag:
            raise Exception("UniTensor.Reshape", "[ERROR] UniTensor.is_diag=True cannot be Reshape.\n" +
                            "[Suggest] Call UniTensor.Todense()")

        if self.braket is not None:
            raise Exception("UniTensor.Reshape",
                            "[ERROR] UniTensor.Reshape can only operate on a [untagged] tensor with regular bonds (BD_REG).")

        if not isinstance(dimer, list):
            raise TypeError("UniTensor.Reshape", "[ERROR] mapper should be an python list.")

        self.Storage = self.Storage.reshape(dimer)

        if new_labels is None:
            new_labels = np.arange(len(dimer))

        self.labels = new_labels
        self.bonds = np.array([Bond(dimer[i]) for i in range(len(dimer))])
        self.rowrank = rowrank

        return self

    def View(self, dimer, rowrank, new_labels=None):
        """
        Return a new view of UniTensor into the shape specified as [dimer], with the first [rowrank] Bonds as bra-bond and other bonds as ket-bond.

        The View() can only operate on a contiguous tensor, otherwise, Contiguous_() or Contiguous() need to be called before the tensor can be viewed. This is the same as pytorch.view().

        [Note] 

            1.View a UniTensor physically re-define the new basis, which construct a new physical definition tensor that has the same element.

            2.View can only operate on an untagged tensor.
            
            3.View requires a contiguous tensor. 

        Args:

            dimer:
                The new shape of the UniTensor. This should be a python list.

            rowrank:
                The number of bonds in row space.

            new_labels:
                The new labels that will be set for new bonds after reshape.

        reture:

            UniTensor

        Example:

            >>> bds_x = [tor10.Bond(6),tor10.Bond(5),tor10.Bond(3)]
            >>> x = tor10.UniTensor(bonds=bds_x, rowrank=1,labels=[4,3,5])
            >>> x.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 3
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 4 ____| 6         5 |____ 3  
                       |             |     
                       |           3 |____ 5  
                       \             /     
                        ------------- 

            >>> x.Permute([0,2,1])
            >>> x.Contiguous_() # this is needed. 
            >>> y = x.View([2,3,5,3],new_labels=[1,2,3,-1],rowrank=2)
            >>> y.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 4
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 1 ____| 2         5 |____ 3  
                       |             |     
                 2 ____| 3         3 |____ -1 
                       \             /     
                        -------------  


        """
        if self.is_symm:
            raise TypeError("UniTensor.View", "[ERROR] Cannot perform View on a symmetry Tensor")

        if self.is_diag:
            raise Exception("UniTensor.View", "[ERROR] UniTensor.is_diag=True cannot be View.\n" +
                            "[Suggest] Call UniTensor.Todense()")

        if not self.is_contiguous():
            raise Exception("UniTensor.View",
                            "[ERROR] UniTensor is not contiguous. Call Contiguous_() or Contiguous() before .View()")

        if self.braket is not None:
            raise Exception("UniTensor.View",
                            "[ERROR] UniTensor.View can only operate on a [untagged] tensor with regular bonds (BD_REG).")

        if not isinstance(dimer, list):
            raise TypeError("UniTensor.View", "[ERROR] mapper should be an python list.")

        new_Storage = copy.deepcopy(self.Storage)

        new_Storage = new_Storage.view(dimer)
        if new_labels is None:
            new_labels = np.arange(len(dimer))

        tmp = UniTensor(bonds=np.array([Bond(dimer[i]) for i in range(len(dimer))]),
                        labels=new_labels,
                        rowrank=rowrank,
                        check=False)

        tmp._mac(torch_tensor=new_Storage)

        return tmp

    def View_(self, dimer, rowrank, new_labels=None):
        """
        Inplace version of View. 
        View UniTensor into the shape specified as [dimer], with the first [rowrank] Bonds as bra-bond and other bonds as ket-bond.

        The View_() can only operate on a contiguous tensor, otherwise, Contiguous_() or Contiguous() need to be called before the tensor can be viewed. This is the inplace version of pytorch.view().
 

        [Note] 

            1.Viewing a UniTensor physically re-define the bra-ket basis space, which construct a new physical definition tensor that has the same element.

            2.Viewing can only operate on an untagged tensor.

            3. View_() requires a contiguous tensor.

        Args:

            dimer:
                The new shape of the UniTensor. This should be a python list.

            rowrank:
                The number of bonds in row space.

            new_labels [option]:
                The new labels that will be set for new bonds after reshape. If not set, the label will be initialize using default enumerate rule.

        Return:

            self
    
        Example:

            >>> bds_x = [tor10.Bond(6),tor10.Bond(5),tor10.Bond(3)]
            >>> x = tor10.UniTensor(bonds=bds_x, rowrank=1,labels=[4,3,5])
            >>> x.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 3
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 4 ____| 6         5 |____ 3  
                       |             |     
                       |           3 |____ 5  
                       \             /     
                        -------------
 
            >>> x.Permute([0,2,1])
            >>> x.Contiguous_() # this is needed
            >>> x.Reshape_([2,3,5,3],new_labels=[1,2,3,-1],rowrank=2)
            >>> x.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 4
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 1 ____| 2         5 |____ 3  
                       |             |     
                 2 ____| 3         3 |____ -1 
                       \             /     
                        -------------


        """
        if self.is_symm:
            raise TypeError("UniTensor.View_()", "[ERROR] Cannot perform View_ on a symmetry Tensor")

        if self.is_diag:
            raise Exception("UniTensor.View_()", "[ERROR] UniTensor.is_diag=True cannot be View_.\n" +
                            "[Suggest] Call UniTensor.Todense()")

        if self.braket is not None:
            raise Exception("UniTensor.View_()",
                            "[ERROR] UniTensor.View_ can only operate on a [untagged] tensor with regular bonds (BD_REG).")

        if not isinstance(dimer, list):
            raise TypeError("UniTensor.View_()", "[ERROR] mapper should be an python list.")

        self.Storage = self.Storage.view(dimer)

        if new_labels is None:
            new_labels = np.arange(len(dimer))

        self.labels = new_labels
        self.bonds = np.array([Bond(dimer[i]) for i in range(len(dimer))])
        self.rowrank = rowrank

        return self

    ## Symmetric Tensor function
    def GetTotalQnums(self, physical=False):
        """
        Return two combined bond objects that has the information for the total qnums at bra and ket bonds.

        Args:

            physical [default: False]:

                Return the physical total qnums.
                
                If True, the return qnums_brabonds will be the physical qnums of all bonds tagged by BD_BRA, and qnums_ketbonds will be the physical qnums of all bonds tagged by BD_KET. 
    
                If False, the return qnums will be the qnums of all bonds in row-space, the mismatch bond will have reversed qnums upon combined. This will match the layout of current blocks. 


        Return:
            qnums_brabonds, qnums_ketbonds:

            qnums_brabonds:
                a tor10.Bond, the combined bra-bond

            qnums_ketbonds:
                a tor10.Bond, the combined ket-bond.


        Example:

            * Multiple Symmetry::

                ## multiple Qnum:
                ## U1 x U1 x U1 x U1
                ## U1 = {-2,-1,0,1,2}
                ## U1 = {-1,1}
                ## U1 = {0,1,2,3}
                bd_sym_1 = tor10.Bond(3,tor10.BD_KET,qnums=[[0, 2, 1, 0],
                                                            [1, 1,-1, 1],
                                                            [2,-1, 1, 0]])
                bd_sym_2 = tor10.Bond(4,tor10.BD_KET,qnums=[[-1, 0,-1, 3],
                                                            [ 0, 0,-1, 2],
                                                            [ 1, 0, 1, 0],
                                                            [ 2,-2,-1, 1]])
                bd_sym_3 = tor10.Bond(2,tor10.BD_BRA,qnums=[[-4, 3, 0,-1],
                                                            [ 1, 1, -2,3]])

                sym_T = tor10.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],rowrank=2,labels=[1,2,3],dtype=torch.float64)
            >>> sym_T.Pring_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 3
            has_symmetry: True
            on device     : cpu
            braket_form : True
                  |ket>               <bra| 
                       ---------------      
                       |             |     
                 1 > __| 3         2 |__ < 3  
                       |             |     
                 2 > __| 4           |        
                       |             |     
                       --------------- 

 
            >>> tqin, tqout = sym_T.GetTotalQnums()
            >>> print(tqin)
            Dim = 12 |
            KET     : U1::  +4 +3 +2 +1 +3 +2 +1 +0 +2 +1 +0 -1
                      U1::  -3 -1 -1 -1 -1 +1 +1 +1 +0 +2 +2 +2
                      U1::  +0 +2 +0 +0 -2 +0 -2 -2 +0 +2 +0 +0
                      U1::  +1 +0 +2 +3 +2 +1 +3 +4 +1 +0 +2 +3

            >>> print(tqout)
            Dim = 2 |
            BRA     : U1::  +1 -4
                      U1::  +1 +3
                      U1::  -2 +0
                      U1::  +3 -1


            >>> sym_T.SetRowRank(1)
            >>> sym_T.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 3
            has_symmetry: True
            on device     : cpu
            braket_form : False
                  |ket>               <bra| 
                       ---------------      
                       |             |     
                 1 > __| 3         4 |__*> 2  
                       |             |     
                       |           2 |__ < 3  
                       |             |   
                       --------------- 

            >>> tqin2,tqout2 =  sym_T.GetTotalQnums()
            >>> print(tqin2)
            Dim = 2 |
            KET     : U1::  -1 +1
                      U1::  -2 +1
                      U1::  -1 -2
                      U1::  +2 +3

            >>> print(tqout2)
            Dim = 8 |
            BRA     : U1::  -1 -6 +0 -5 +1 -4 +2 -3
                      U1::  +3 +5 +1 +3 +1 +3 +1 +3
                      U1::  -1 +1 -3 -1 -1 +1 -1 +1
                      U1::  +2 -2 +3 -1 +1 -3 +0 -4
 
            >>> tqin2_phy, tqout2_phy = sym_T.GetTotalQnums(physical=True)
            >>> print(tqin2_phy)
            Dim = 12 |
            KET     : U1::  +4 +3 +2 +1 +3 +2 +1 +0 +2 +1 +0 -1
                      U1::  -3 -1 -1 -1 -1 +1 +1 +1 +0 +2 +2 +2
                      U1::  +0 +2 +0 +0 -2 +0 -2 -2 +0 +2 +0 +0
                      U1::  +1 +0 +2 +3 +2 +1 +3 +4 +1 +0 +2 +3

            >>> print(tqout2_phy)
            Dim = 2 |
            BRA     : U1::  +1 -4
                      U1::  +1 +3
                      U1::  -2 +0
                      U1::  +3 -1
          
            >>> print(tqin2 == tqin  ) ## this should be False
            False

            >>> print(tqout2 == tqout) ## this should be False
            False

            >>> print(tqin2_phy == tqin) ## this should be true
            True
        
            >>> print(tqout2_phy == tqout) ## this should be true
            True

        """
        if not self.is_symm:
            raise TypeError("UniTensor.GetTotalQnums", "[ERROR] GetTotal Qnums from a non-symm tensor")

        # if (self.rowrank==0) or (self.rowrank==len(self.bonds)):
        #    raise Exception("UniTensor.GetTotalQnums","[ERROR] The TN symmetry structure is incorrect, without either any in-bond or any-outbond")
        if physical:
            # virtual_cb-in
            cb_inbonds = copy.deepcopy(self.bonds[np.argwhere(self.braket == BondType[BD_KET]).flatten()])
            in_all = cb_inbonds[0]
            if len(cb_inbonds) > 1:
                in_all.combine(cb_inbonds[1:])

            cb_outbonds = copy.deepcopy(self.bonds[np.argwhere(self.braket == BondType[BD_BRA]).flatten()])
            out_all = cb_outbonds[0]
            if len(cb_outbonds) > 1:
                out_all.combine(cb_outbonds[1:])
        else:
            # virtual_cb-in
            cb_inbonds = copy.deepcopy(self.bonds[:self.rowrank]) * self.braket[:self.rowrank] * BondType[BD_KET]
            in_all = cb_inbonds[0]
            if len(cb_inbonds) > 1:
                in_all.combine(cb_inbonds[1:])
            cb_outbonds = copy.deepcopy(self.bonds[self.rowrank:]) * self.braket[self.rowrank:] * BondType[BD_BRA]
            out_all = cb_outbonds[0]
            if len(cb_outbonds) > 1:
                out_all.combine(cb_outbonds[1:])

        in_all.bondType = BD_KET
        out_all.bondType = BD_BRA

        return in_all, out_all

    def GetValidQnums(self, physical=False, return_shape=False):
        """
            Return the quantum number set that has a valid block.

            Args:
                
                physical [default: False]:
                    
                    if set to True, return the unique quantum number sets defined by BD_BRA and BD_KET.

                    The return 2D array has shape (# of blocks,qnum set)

                return_shape [default: False]:
            
                    if set to True, return a 2D array with shape (# of blocks, size of each block)

            Return 
                
                if return_shape == False: return [qnum sets, 2D ndarray]  
                if return_shape == True : return [qnum sets, 2D ndarray], [shape (2D ndarray)]
                    
                    

        """
        if physical:
            b_tqin, b_tqout = self.GetTotalQnums(physical=True)
            tqin_uni = b_tqin.GetUniqueQnums()
            tqout_uni = b_tqout.GetUniqueQnums()
            comm = _fx_GetCommRows(tqin_uni, tqout_uni)
            shap = []
            if return_shape:
                for q in comm:
                    shap.append(np.array([b_tqin.GetDegeneracy(*q), b_tqout.GetDegeneracy(*q)]))
                return comm, np.array(shap)
            else:
                return comm
        else:
            comm = copy.deepcopy(self._block_qnums)
            if return_shape:
                b_tqin, b_tqout = self.GetTotalQnums(physical=False)
                shap = []
                for q in comm:
                    shap.append(np.array([b_tqin.GetDegeneracy(*q), b_tqout.GetDegeneracy(*q)]))
                return comm, np.array(shap)
            else:
                return comm

    def PutBlock(self, block, *qnum):
        """
        Put the block into the UniTensor. If the UniTensor is symmetry tensor, the block should be specify by the quantum number. 
       
        Args:
            block:
                A UniTensor with rank-2
            
            *qnum:
                The quantum number set that specify the block.

        
        """
        if not isinstance(block, self.__class__):
            raise TypeError("[ERROR] PutBlock can only accept a untagged UniTensor ")

        ## Note, block should be a UniTensor:
        if block.braket is not None:
            raise Exception("[ERROR] PutBlock can only accept a untagged UniTensor ")

        if not self.is_symm:
            ## check:

            if self.is_diag:
                if not block.is_diag:
                    raise Exception(
                        "[ERROR] PutBlock for a is_diag=True tensor can only accept a block with is_diag=True")

            if self.rowrank == 0:
                curr_shape_2d = torch.Size([self.Storage.numel()])
                if block.shape != curr_shape_2d:
                    raise Exception("[ERROR] the shape of input Block", block.shape,
                                    "does not match the shape of current block", curr_shape_2d)

            elif len(self.bonds) - self.rowrank == 0:
                curr_shape_2d = torch.Size([self.Storage.numel()])
                if block.shape != curr_shape_2d:
                    raise Exception("[ERROR] the shape of input Block", block.shape,
                                    "does not match the shape of current block", curr_shape_2d)
            else:
                curr_shape_2d = torch.Size([np.prod([x.dim for x in self.bonds[:self.rowrank]]),
                                            np.prod([x.dim for x in self.bonds[self.rowrank:]])])
                if block.shape != curr_shape_2d:
                    raise Exception("[ERROR] the shape of input Block", block.shape,
                                    "does not match the shape of current block", curr_shape_2d)

            ## memcpy low-lv-api
            #self.Storage.storage().copy_(block.Storage.storage())
            shp = self.Storage.shape
            self.Storage = block.Storage.clone().reshape(shp)
            # raise Exception("[Warning] PutBlock cannot be use for non-symmetry TN. Use SetElem instead.")

        else:

            # raise Exception("Developing")

            if len(qnum) != self.bonds[0].nsym:
                raise ValueError("UniTensor.PutBlock", "[ERROR] The quantum numbers do not match the number of types.")

            ## check contiguous:
            if self._contiguous:
                is_set = False
                ## search if the tn has block of that qnums:
                for s in range(len(self._block_qnums)):
                    if (np.array(qnum) == self._block_qnums[s]).all():
                        ##check if shape is correct:
                        if self.Storage[s].shape != block.shape:
                            raise TypeError("UniTensor.PutBlock", "[ERROR] the input block with shape", block.shape,
                                            "does not match the current block's shape", self.Storage[s].shape)
                        #self.Storage[s].storage().copy_(block.Storage.storage())
                        shp = self.Storage[s].shape
                        self.Storage[s] = block.Storage.clone().reshape(shp)
                        is_set = True
                        break
                if not is_set:
                    raise TypeError("UniTensor.PutBlock", "[ERROR] no block has qnums:", qnum)

            else:

                ## search the current valid blocks :
                is_set = False
                for s in range(len(self._block_qnums)):
                    if (np.array(qnum) == self._block_qnums[s]).all():
                        ## get Nrowrank for the memory
                        old_rowrank = len(self._Ket_invmapper_blks[0][0])

                        accu_off = []
                        tmp = 1
                        for i in range(len(self.bonds)):
                            accu_off.append(tmp)
                            tmp *= self.bonds[-1 - i].dim
                        accu_off = np.array(accu_off[::-1])

                        new_accu_off_in = (accu_off[:self.rowrank] / accu_off[self.rowrank - 1]).astype(np.int)
                        new_accu_off_out = accu_off[self.rowrank:]
                        del accu_off

                        ## copy from the right address.
                        b_tqin, b_tqout = self.GetTotalQnums(physical=False)
                        idx_in = np.argwhere((b_tqin.qnums == self._block_qnums[s]).all(axis=1)).flatten()
                        idx_out = np.argwhere((b_tqout.qnums == self._block_qnums[s]).all(axis=1)).flatten()

                        ## interface
                        new_Ket_invmapper_blks = _fx_decompress_idx(idx_in, new_accu_off_in)
                        # self._Ket_mapper_blks[idx_in,0] = b
                        # self._Ket_mapper_blks[idx_in,1] = np.arange(len(idx_in)).astype(np.int)

                        ## interface
                        new_Bra_invmapper_blks = _fx_decompress_idx(idx_out, new_accu_off_out)
                        # self._Bra_mapper_blks[idx_out,0] = b
                        # self._Bra_mapper_blks[idx_out,1] = np.arange(len(idx_out)).astype(np.int)

                        ## Get element only for this block from the right memory place:
                        # old_rowrank = self._Ket_invmapper_blks[0].
                        for i in range(len(idx_in)):
                            for j in range(len(idx_out)):
                                newidx = np.concatenate((new_Ket_invmapper_blks[i], new_Bra_invmapper_blks[j]))
                                oldidx = newidx[self._inv_mapper]

                                old_row = int(np.sum(self._accu_off_in * oldidx[:old_rowrank]))
                                old_col = int(np.sum(self._accu_off_out * oldidx[old_rowrank:]))

                                b_id_in = self._Ket_mapper_blks[old_row]
                                b_id_out = self._Bra_mapper_blks[old_col]

                                if b_id_in[0] != b_id_out[0]:
                                    raise Exception("[ERROR] internal FATAL")

                                if b_id_in[0] >= 0 and b_id_out[0] >= 0:
                                    self.Storage[b_id_in[0]][b_id_in[1], b_id_out[1]] = block.Storage[i, j].clone()
                                else:
                                    print("[unphys pos]")

                        is_set = True
                        break

                        ## if there is no block with qnum:
                if not is_set:
                    raise TypeError("UniTensor.PutBlock", "[ERROR] No block has qnums:", qnum)

    def GetBlock(self, *qnum):
        """
        Return the Block specify by the quantum number(s). If the UniTensor is non-symmetry, return self.

        Args:
            *qnum:
                The quantum number(s). Note that when get-block on a High-rank tensor, the quantum number represent the total quantum number of all the in(out)-bonds.

        Return:
            * UniTensor, rank-2 (for symmetry tensor)
            * a new rank-2 flattened UniTensor (for non-symmetry tensor)

        Example:
            * Single Symmetry::

                bd_sym_1 = tor10.Bond(3,tor10.BD_KET,qnums=[[0],[1],[2]])
                bd_sym_2 = tor10.Bond(4,tor10.BD_KET,qnums=[[-1],[2],[0],[2]])
                bd_sym_3 = tor10.Bond(5,tor10.BD_BRA,qnums=[[4],[2],[2],[5],[1]])
                sym_T = tor10.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],rowrank=2,labels=[10,11,12],dtype=torch.float64)

            >>> sym_T.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 3
            has_symmetry: True
            on device     : cpu
            braket_form : True
                  |ket>               <bra| 
                       ---------------      
                       |             |     
                10 > __| 3         5 |__ < 12 
                       |             |     
                11 > __| 4           |        
                       |             |     
                       ---------------  

            >>> q_in, q_out = sym_T.GetTotalQnums()
            >>> print(q_in)
            Dim = 12 |
            KET     : U1::  +4 +4 +2 +1 +3 +3 +1 +0 +2 +2 +0 -1

            >>> print(q_out)
            Dim = 5 |
            BRA     : U1::  +5 +4 +2 +2 +1

            >>> bk2 = sym_T.GetBlock(2)
            >>> bk2.Print_diagram()
            -----------------------
            tensor Name : 
            tensor Rank : 2
            has_symmetry: False
            on device     : cpu
            is_diag       : False
                        -------------      
                       /             \     
                 0 ____| 3         2 |____ 1  
                       \             /     
                        -------------  
 
            >>> print(bk2)
            Tensor name: 
            is_diag    : False
            tensor([[0., 0.],
                    [0., 0.],
                    [0., 0.]], dtype=torch.float64)

            * Multiple Symmetry::

                ## multiple Qnum:
                ## U1 x U1 x U1 x U1
                bd_sym_1 = tor10.Bond(3,tor10.BD_KET,qnums=[[0, 2, 1, 0],
                                                            [1, 1,-1, 1],
                                                            [2,-1, 1, 0]])
                bd_sym_2 = tor10.Bond(4,tor10.BD_KET,qnums=[[-1, 0,-1, 3],
                                                            [ 0, 0,-1, 2],
                                                            [ 1, 0, 1, 0],
                                                            [ 2,-2,-1, 1]])
                bd_sym_3 = tor10.Bond(2,tor10.BD_BRA,qnums=[[-1,-2,-1,2],
                                                            [ 1, 1, -2,3]])

                sym_T = tor10.UniTensor(bonds=[bd_sym_1,bd_sym_2,bd_sym_3],rowrank=2,labels=[1,2,3],dtype=torch.float64)

            >>> tqin, tqout = sym_T.GetTotalQnums()
            >>> print(tqin)
            Dim = 12 |
            KET     : U1::  +4 +3 +2 +1 +3 +2 +1 +0 +2 +1 +0 -1
                      U1::  -3 -1 -1 -1 -1 +1 +1 +1 +0 +2 +2 +2
                      U1::  +0 +2 +0 +0 -2 +0 -2 -2 +0 +2 +0 +0
                      U1::  +1 +0 +2 +3 +2 +1 +3 +4 +1 +0 +2 +3

            >>> print(tqout)
            Dim = 2 |
            BRA     : U1::  +1 -1
                      U1::  +1 -2
                      U1::  -2 -1
                      U1::  +3 +2

            >>> block_1123 = sym_T.GetBlock(1,1,-2,3)
            >>> print(block_1123)
            Tensor name: 
            is_diag    : False
            tensor([[0.]], dtype=torch.float64)




        """
        if not self.is_symm:

            if self.is_diag:
                bds = [Bond(self.Storage.bonds[0].dim), Bond(self.Storage.bonds[0].dim)]
                tmp = UniTensor(bonds=bds, rowrank=1, check=False, is_diag=True)
                tmp._mac(torch_tensor=self.Storage.clone())
                return tmp
            else:

                if self.rowrank == 0:
                    bds = [Bond(self.Storage.numel())]
                    tmp = UniTensor(bonds=bds, rowrank=0, check=False)
                    tmp._mac(torch_tensor=self.Storage.flatten())
                    return tmp

                elif len(self.bonds) - self.rowrank == 0:
                    bds = [Bond(self.Storage.numel())]
                    tmp = UniTensor(bonds=bds, rowrank=1, check=False)
                    tmp._mac(torch_tensor=self.Storage.flatten())
                    return tmp
                else:
                    bds = [Bond(np.prod([x.dim for x in self.bonds[:self.rowrank]])),
                           Bond(np.prod([x.dim for x in self.bonds[self.rowrank:]]))]

                    tmp = UniTensor(bonds=bds, rowrank=1, check=False)
                    tmp._mac(torch_tensor=self.Storage.reshape(bds[0].dim, -1))
                    return tmp
        else:
            # raise Exception("[Developing]")

            # if not self.is_braket:
            #    raise Exception("[ERROR] Can only get block from a symmetry Tensor in it's bra-ket form\n   Suggestion: call to_braket_form() or manually permute the tensor to the braket form. before get-block")

            if len(qnum) != self.bonds[0].nsym:
                raise ValueError("UniTensor.GetBlock", "[ERROR] The qnumtum numbers not match the number of type.")

            ## check contiguous:
            if self._contiguous:
                ## search if the tn has block of that qnums:
                for s in range(len(self._block_qnums)):
                    if (np.array(qnum) == self._block_qnums[s]).all():
                        tmp = UniTensor(bonds=[Bond(self.Storage[s].shape[0]), Bond(self.Storage[s].shape[1])],
                                        rowrank=1,
                                        check=False)
                        tmp._mac(torch_tensor=self.Storage[s].clone())
                        return tmp
                ## if there is no block with qnum:
                raise TypeError("UniTensor.GetBlock", "[ERROR] No block has qnums:", qnum)
            else:
                ## search the current valid blocks :
                for s in range(len(self._block_qnums)):
                    if (np.array(qnum) == self._block_qnums[s]).all():
                        ## get Nrowrank for the memory
                        old_rowrank = len(self._Ket_invmapper_blks[0][0])

                        accu_off = []
                        tmp = 1
                        for i in range(len(self.bonds)):
                            accu_off.append(tmp)
                            tmp *= self.bonds[-1 - i].dim
                        accu_off = np.array(accu_off[::-1])

                        new_accu_off_in = (accu_off[:self.rowrank] / accu_off[self.rowrank - 1]).astype(np.int)
                        new_accu_off_out = accu_off[self.rowrank:]
                        del accu_off

                        ## copy from the right address.
                        b_tqin, b_tqout = self.GetTotalQnums(physical=False)
                        idx_in = np.argwhere((b_tqin.qnums == self._block_qnums[s]).all(axis=1)).flatten()
                        idx_out = np.argwhere((b_tqout.qnums == self._block_qnums[s]).all(axis=1)).flatten()

                        ## Create only the block:
                        Block = torch.zeros((len(idx_in), len(idx_out)), device=self.device, dtype=self.dtype)

                        ## interface
                        new_Ket_invmapper_blks = _fx_decompress_idx(idx_in, new_accu_off_in)
                        # self._Ket_mapper_blks[idx_in,0] = b
                        # self._Ket_mapper_blks[idx_in,1] = np.arange(len(idx_in)).astype(np.int)

                        ## interface
                        new_Bra_invmapper_blks = _fx_decompress_idx(idx_out, new_accu_off_out)
                        # self._Bra_mapper_blks[idx_out,0] = b
                        # self._Bra_mapper_blks[idx_out,1] = np.arange(len(idx_out)).astype(np.int)

                        ## Get element only for this block from the right memory place:
                        # old_rowrank = self._Ket_invmapper_blks[0].
                        for i in range(len(idx_in)):
                            for j in range(len(idx_out)):
                                newidx = np.concatenate((new_Ket_invmapper_blks[i], new_Bra_invmapper_blks[j]))
                                oldidx = newidx[self._inv_mapper]

                                old_row = int(np.sum(self._accu_off_in * oldidx[:old_rowrank]))
                                old_col = int(np.sum(self._accu_off_out * oldidx[old_rowrank:]))

                                b_id_in = self._Ket_mapper_blks[old_row]
                                b_id_out = self._Bra_mapper_blks[old_col]

                                ## [DEBUG] >>>
                                if DEBUG:
                                    if b_id_in[0] != b_id_out[0]:
                                        raise Exception("[ERROR] internal FATAL")
                                ## <<<

                                if b_id_in[0] >= 0 and b_id_out[0] >= 0:
                                    Block[i, j] = self.Storage[b_id_in[0]][b_id_in[1], b_id_out[1]]
                                else:
                                    ## [DEBUG] >>>
                                    if DEBUG:
                                        print("[ERROR] unphys pos!")
                                    ## <<<<

                        tmp = UniTensor(bonds=[Bond(Block.shape[0]), Bond(Block.shape[1])],
                                        check=False,
                                        rowrank=1)
                        tmp._mac(torch_tensor=Block)
                        return tmp
                ## if there is no block with qnum:
                raise TypeError("UniTensor.GetBlock", "[ERROR] No block has qnums:", qnum)

    def torch(self):
        """
        Transform a UniTensor to torch.Tensor. 

            [Note]
                
                1. this cannot be operate on a UniTensor with symmetry.
                2. the return tensor will not share the same memory with the UniTensor.

        Return:
            
            torch.Tensor


        """
        if self.is_symm:
            raise Exception("[ERROR] cannot transform the UniTensor with symmetry to torch.Tensor. GetBlock first.")
        else:
            return self.Storage.clone()

    ## Autograd feature:
    def requires_grad(self, is_grad=None):
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
            bds_x = [tor10.Bond(5),tor10.Bond(5),tor10.Bond(3)]
            x = tor10.UniTensor(bonds=bds_x, rowrank=2, labels=[4,3,5])


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
            if self.is_symm:
                return self.Storage[0].requires_grad
            else:
                return self.Storage.requires_grad
        else:
            if self.is_symm:
                for s in range(len(self.Storage)):
                    self.Storage[s].requires_grad_(bool(is_grad))
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

            >>> x = tor10.UniTensor(bonds=[tor10.Bond(2),tor10.Bond(2)],rowrank=1,requires_grad=True)
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

            >>> out = tor10.Mean(y)
            >>> print(out)
            Tensor name:
            is_diag    : False
            tensor(16., dtype=torch.float64, grad_fn=<MeanBackward1>)

            >>> out.backward()
            >>> print(x.grad())
            Tensor name:
            is_diag    : False
            tensor([[2., 2.],
                    [2., 2.]], dtype=torch.float64)

        """
        if not self.requires_grad():
            return None
        else:
            if self.is_symm:

                tmp = UniTensor(bonds=self.bonds,
                                labels=self.labels,
                                rowrank=self.rowrank,
                                check=False)
                tmp._mac(torch_tensor=[self.Storage[s].grad for s in range(len(self.Storage))],
                         braket=self.braket,
                         sym_mappers=(self._mapper, self._inv_mapper,
                                      self._Ket_mapper_blks, self._Ket_invmapper_blks,
                                      self._Bra_mapper_blks, self._Bra_invmapper_blks,
                                      self._contiguous, self._accu_off_in, self._accu_off_out))
                # raise Exception("Developing")
            else:
                tmp = UniTensor(bonds=copy.deepcopy(self.bonds),
                                rowrank=self.rowrank,
                                check=False)
                tmp._mac(torch_tensor=self.Storage.grad)
            return tmp

    def backward(self):
        """
        Backward the gradient flow in the constructed autograd graph. This is the same as torch.Tensor.backward
        """
        if self.is_symm:
            for s in range(len(self.Storage)):
                self.Storage[s].backward()

        else:
            self.Storage.backward()

    def detach(self):
        """
        Detach the current tensor from the current graph, making it a leaf. This is the same as torch.Tensor.detach_()

        Return:
            self
        """
        if self.is_symm:
            for s in range(len(self.Storage)):
                self.Storage[s].detach_()
        else:
            self.Storage.detach_()
        return self


###############################################################
#
# Action function
#
##############################################################
## I/O
def Save(a, filename):
    """
    Save a UniTensor to the file

    Args:
        a:
            The UniTensor that to be saved.

        filename:
            The saved file path

    Example:
    ::
        a = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(4)],rowrank=1)
        tor10.Save(a,"a.uniT")

    """
    if not isinstance(filename, str):
        raise TypeError("Save", "[ERROR] Invalid filename.")
    if not isinstance(a, UniTensor):
        raise TypeError("Save", "[ERROR] input must be the UniTensor")
    f = open(filename, "wb")
    pkl.dump(a, f)
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
        a = tor10.Load("a.uniT")

    """
    if not isinstance(filename, str):
        raise TypeError("UniTensor.Save", "[ERROR] Invalid filename.")
    if not os.path.exists(filename):
        raise Exception("UniTensor.Load", "[ERROR] file not exists")

    f = open(filename, 'rb')
    tmp = pkl.load(f)
    f.close()
    if not isinstance(tmp, UniTensor):
        raise TypeError("Load", "[ERROR] loaded object is not the UniTensor")

    return tmp


def Contract(a, b):
    """
        Contract two tensors with the same labels.

        1. two tensors must be the same type, if "a" is a symmetry/untagged/tagged tensor, "b" must also be a symmetry/untagged/tagged tensor.
        2. When contract two symmetry tensor, the bonds that to be contracted must have the same qnums.

        3. For tagged tensor, Each bra-bond can only contract with ket-bond, in terms of physical meaning, this means the contract traceing out the matched bra-ket.

        [Note] the argument "a" and "b" tensor defines the order of the out-come bond. After contract,  the order of remaining bonds (both in-bond(bra) and out-bond(ket)) that appears in the new-tensor will follows the rule: a's in-bond will appears first, then the b's in-bond; a's out-bond will appears first, then b's out-bond (see example in below)


        Args:
            a:
                UniTensor

            b:
                UniTensor


        Return:
            UniTensor

        Example:
        ::
            x = tor10.UniTensor(bonds=[tor10.Bond(5),tor10.Bond(2),tor10.Bond(4),tor10.Bond(3)], rowrank=2,labels=[6,1,7,8])
            y = tor10.UniTensor(bonds=[tor10.Bond(4),tor10.Bond(2),tor10.Bond(3),tor10.Bond(6)], rowrank=2,labels=[7,2,10,9])


        >>> x.Print_diagram()
        -----------------------
        tensor Name : 
        tensor Rank : 4
        has_symmetry: False
        on device     : cpu
        is_diag       : False
                    -------------      
                   /             \     
             6 ____| 5         4 |____ 7  
                   |             |     
             1 ____| 2         3 |____ 8  
                   \             /     
                    -------------   

        >>> y.Print_diagram()
        -----------------------
        tensor Name : 
        tensor Rank : 4
        has_symmetry: False
        on device     : cpu
        is_diag       : False
                    -------------      
                   /             \     
             7 ____| 4         3 |____ 10 
                   |             |     
             2 ____| 2         6 |____ 9  
                   \             /     
                    -------------  

        >>> c = tor10.Contract(x,y)
        >>> c.Print_diagram()
        -----------------------
        tensor Name : 
        tensor Rank : 6
        has_symmetry: False
        on device     : cpu
        is_diag       : False
                    -------------      
                   /             \     
             6 ____| 5         3 |____ 8  
                   |             |     
             1 ____| 2         3 |____ 10 
                   |             |     
             2 ____| 2         6 |____ 9  
                   \             /     
                    -------------  

        >>> d = tor10.Contract(y,x)
        >>> d.Print_diagram()
        -----------------------
        tensor Name : 
        tensor Rank : 6
        has_symmetry: False
        on device     : cpu
        is_diag       : False
                    -------------      
                   /             \     
             2 ____| 2         3 |____ 10 
                   |             |     
             6 ____| 5         6 |____ 9  
                   |             |     
             1 ____| 2         3 |____ 8  
                   \             /     
                    -------------  


        Note that you can also contract for UniTensor with symmetry, even when they are not in the bra-ket form. As long as the quantum number on the to-be-contract bonds and the bond type matches (bra can only contract with ket)
        ::
            bd_sym_1a = tor10.Bond(3,tor10.BD_KET,qnums=[[0],[1],[2]])
            bd_sym_2a = tor10.Bond(4,tor10.BD_KET,qnums=[[-1],[2],[0],[2]])
            bd_sym_3a = tor10.Bond(5,tor10.BD_BRA,qnums=[[4],[2],[-1],[5],[1]])

            bd_sym_1b = tor10.Bond(3,tor10.BD_BRA,qnums=[[0],[1],[2]])
            bd_sym_2b = tor10.Bond(4,tor10.BD_BRA,qnums=[[-1],[2],[0],[2]])
            bd_sym_3b = tor10.Bond(7,tor10.BD_KET,qnums=[[1],[3],[-2],[2],[2],[2],[0]])

            sym_A = tor10.UniTensor(bonds=[bd_sym_1a,bd_sym_2a,bd_sym_3a],rowrank=2,labels=[10,11,12])
            sym_B = tor10.UniTensor(bonds=[bd_sym_2b,bd_sym_1b,bd_sym_3b],rowrank=1,labels=[11,10,7])


        >>> sym_A.Print_diagram()
        -----------------------
        tensor Name : 
        tensor Rank : 3
        has_symmetry: True
        on device     : cpu
        braket_form : True
              |ket>               <bra| 
                   ---------------      
                   |             |     
            10 > __| 3         5 |__ < 12 
                   |             |     
            11 > __| 4           |        
                   |             |     
                   ---------------  


        >>> sym_B.Print_diagram()
        -----------------------
        tensor Name : 
        tensor Rank : 3
        has_symmetry: True
        on device     : cpu
        braket_form : False
              |ket>               <bra| 
                   ---------------      
                   |             |     
            11 <*__| 4         3 |__ < 10 
                   |             |     
                   |           7 |__*> 7  
                   |             |     
                   ---------------  

        >>> sym_AB = tor10.Contract(sym_A,sym_B)
        >>> sym_BA = tor10.Contract(sym_B,sym_A)
        >>> sym_AB.Print_diagram()
        -----------------------
        tensor Name : 
        tensor Rank : 2
        has_symmetry: True
        on device     : cpu
        braket_form : False
              |ket>               <bra| 
                   ---------------      
                   |             |     
            12 <*__| 5         7 |__*> 7  
                   |             |     
                   ---------------  
     
        >>> sym_BA.Print_diagram()
        -----------------------
        tensor Name : 
        tensor Rank : 2
        has_symmetry: True
        on device     : cpu
        braket_form : True
              |ket>               <bra| 
                   ---------------      
                   |             |     
             7 > __| 7         5 |__ < 12 
                   |             |     
                   --------------- 

    """
    if isinstance(a, UniTensor) and isinstance(b, UniTensor):

        ## check:
        if (a.is_symm != b.is_symm) or ((a.braket is None) != (b.braket is None)):
            raise TypeError("Contract(a,b)", "[ERROR] the tensors should be the same type to be contracted")

        ## get same vector:
        same, a_ind, b_ind = np.intersect1d(a.labels, b.labels, return_indices=True)

        aind_no_combine = np.setdiff1d(np.arange(len(a.labels)), a_ind)
        bind_no_combine = np.setdiff1d(np.arange(len(b.labels)), b_ind)

        ## master switch
        if a.is_symm:
            ## contract symm tensor > 
            if len(same):

                for i in range(len(a_ind)):
                    if not a.bonds[a_ind[i]].qnums.all() == b.bonds[b_ind[i]].qnums.all():
                        raise ValueError("Contact(a,b)", "[ERROR] contract Bonds that has qnums mismatch.")

                if False in np.unique((a.braket[a_ind] + b.braket[b_ind]) == 0):
                    raise Exception("Contract(a,b)", "[ERROR] bra-bond can only contract with ket-bond")

                tmpa = copy.deepcopy(a)
                tmpb = copy.deepcopy(b)
                tmpa.Permute(np.append(aind_no_combine, a_ind), rowrank=len(a.labels) - len(a_ind),
                             by_label=False).Contiguous_()
                tmpb.Permute(np.append(b_ind, bind_no_combine), rowrank=len(b_ind), by_label=False).Contiguous_()

                # tmpa.Print_diagram()
                # print(tmpa)
                # print(tmpa.GetValidQnums(return_shape=True))
                # tmpb.Print_diagram()
                # print(tmpb)
                # print(tmpb.GetValidQnums(return_shape=True))
                # tmpa._Bra_mapper_blks

                aQ = tmpa.GetValidQnums()
                bQ = tmpb.GetValidQnums()

                out = UniTensor(bonds=np.append(tmpa.bonds[:tmpa.rowrank], tmpb.bonds[tmpb.rowrank:]),
                                labels=np.append(tmpa.labels[:tmpa.rowrank], tmpb.labels[tmpb.rowrank:]),
                                dtype=a.dtype,
                                device=a.device)

                oQ = out.GetValidQnums()

                for obid in range(len(oQ)):
                    ab = None
                    for abid in range(len(aQ)):
                        if (oQ[obid] == aQ[abid]).all():
                            ab = abid
                            break

                    bb = None
                    for bbid in range(len(bQ)):
                        if (oQ[obid] == bQ[bbid]).all():
                            bb = bbid
                            break

                    if (ab is not None) and (bb is not None):
                        out.Storage[obid] = torch.matmul(tmpa.Storage[ab], tmpb.Storage[bb])

                return out

            else:
                ## product!!
                raise Exception("Developing")




        else:
            ## contract non-sym tensor > 

            if len(same):
                if a.is_diag:

                    tmpa = torch.diag(a.Storage).to(a.Storage.device)
                else:
                    tmpa = a.Storage

                if b.is_diag:
                    tmpb = torch.diag(b.Storage).to(b.Storage.device)
                else:
                    tmpb = b.Storage

                if a.braket is None:
                    ## contract untagged 
                    tmp = torch.tensordot(tmpa, tmpb, dims=(a_ind.tolist(), b_ind.tolist()))

                    new_bonds = np.concatenate(
                        [copy.deepcopy(a.bonds[aind_no_combine]), copy.deepcopy(b.bonds[bind_no_combine])])
                    new_io = [(aind_no_combine[x] >= a.rowrank) for x in range(len(aind_no_combine))] + [
                        (bind_no_combine[x] >= b.rowrank) for x in range(len(bind_no_combine))]
                    new_labels = np.concatenate(
                        [copy.copy(a.labels[aind_no_combine]), copy.copy(b.labels[bind_no_combine])])

                    new_io = np.array(new_io)
                    # print(new_io)
                    if len(new_bonds) > 0:
                        mapper = np.argsort(new_io)
                        new_bonds = new_bonds[mapper]
                        new_labels = new_labels[mapper]
                        tmp = tmp.permute(*mapper)

                    out = UniTensor(bonds=new_bonds,
                                    labels=new_labels,
                                    rowrank=len(np.argwhere(new_io == 0)),
                                    check=False)
                    out._mac(torch_tensor=tmp)

                    return out
                else:
                    ## tagged
                    if False in np.unique((a.braket[a_ind] + b.braket[b_ind]) == 0):
                        raise Exception("Contract(a,b)", "[ERROR] in-bond(bra) can only contract with out-bond (ket)")

                    tmp = torch.tensordot(tmpa, tmpb, dims=(a_ind.tolist(), b_ind.tolist()))

                    new_bonds = np.concatenate(
                        [copy.deepcopy(a.bonds[aind_no_combine]), copy.deepcopy(b.bonds[bind_no_combine])])
                    new_io = [(aind_no_combine[x] >= a.rowrank) for x in range(len(aind_no_combine))] + [
                        (bind_no_combine[x] >= b.rowrank) for x in range(len(bind_no_combine))]
                    new_labels = np.concatenate(
                        [copy.copy(a.labels[aind_no_combine]), copy.copy(b.labels[bind_no_combine])])
                    new_braket = np.concatenate(
                        [copy.copy(a.braket[aind_no_combine]), copy.copy(b.braket[bind_no_combine])])

                    new_io = np.array(new_io)
                    # print(new_io)
                    if len(new_bonds) > 0:
                        mapper = np.argsort(new_io)
                        new_bonds = new_bonds[mapper]
                        new_labels = new_labels[mapper]
                        new_braket = new_braket[mapper]
                        tmp = tmp.permute(*mapper)

                    out = UniTensor(bonds=new_bonds,
                                    labels=new_labels,
                                    rowrank=len(np.argwhere(new_io == 0)),
                                    check=False)
                    out._mac(braket=new_braket, torch_tensor=tmp)

                    return out
            else:
                ## product!!
                if a.is_diag:
                    tmpa = torch.diag(a.Storage)
                else:
                    tmpa = a.Storage

                if b.is_diag:
                    tmpb = torch.diag(b.Storage)
                else:
                    tmpb = b.Storage

                if a.braket is None:
                    ## untagged 
                    tmp = torch.tensordot(tmpa, tmpb, dims=0)
                    new_bonds = np.concatenate([copy.deepcopy(a.bonds), copy.deepcopy(b.bonds)])
                    new_labels = np.concatenate([copy.copy(a.labels), copy.copy(b.labels)])
                    new_io = [(x >= a.rowrank) for x in range(len(a.bonds))] + [(x >= b.rowrank) for x in
                                                                                range(len(b.bonds))]

                    if len(new_bonds) > 0:
                        mapper = np.argsort(new_io)
                        new_bonds = new_bonds[mapper]
                        new_labels = new_labels[mapper]
                        tmp = tmp.permute(*mapper)

                    out = UniTensor(bonds=new_bonds,
                                    labels=new_labels,
                                    rowrank=a.rowrank + b.rowrank,
                                    check=False)
                    out._mac(torch_tensor=tmp)
                    return out

                else:
                    ## tagged
                    tmp = torch.tensordot(tmpa, tmpb, dims=0)
                    new_bonds = np.concatenate([copy.deepcopy(a.bonds), copy.deepcopy(b.bonds)])
                    new_labels = np.concatenate([copy.copy(a.labels), copy.copy(b.labels)])
                    new_braket = np.concatenate([copy.copy(a.braket), copy.copy(b.braket)])
                    new_io = [(x >= a.rowrank) for x in range(len(a.bonds))] + [(x >= b.rowrank) for x in
                                                                                range(len(b.bonds))]

                    if len(new_bonds) > 0:
                        mapper = np.argsort(new_io)
                        new_bonds = new_bonds[mapper]
                        new_labels = new_labels[mapper]
                        new_braket = new_braket[mapper]
                        tmp = tmp.permute(*mapper)

                    out = UniTensor(bonds=new_bonds,
                                    labels=new_labels,
                                    rowrank=a.rowrank + b.rowrank,
                                    check=False)
                    out._mac(braket=new_braket,
                             torch_tensor=tmp)
                    return out
        if len(same):

            # if(a.is_symm):
            #    for i in range(len(a_ind)):
            #        if not a.bonds[a_ind[i]].qnums.all() == b.bonds[b_ind[i]].qnums.all():
            #            raise ValueError("Contact(a,b)","[ERROR] contract Bonds that has qnums mismatch.")

            ## check bra-ket
            if a.braket is None:
                pass
            else:
                if False in np.unique((a.braket[a_ind] + b.braket[b_ind]) == 0):
                    raise Exception("Contract(a,b)", "[ERROR] in-bond(bra) can only contract with out-bond (ket)")

            aind_no_combine = np.setdiff1d(np.arange(len(a.labels)), a_ind)
            bind_no_combine = np.setdiff1d(np.arange(len(b.labels)), b_ind)

            if a.is_diag:
                tmpa = torch.diag(a.Storage).to(a.Storage.device)
            else:
                tmpa = a.Storage

            if b.is_diag:
                tmpb = torch.diag(b.Storage).to(b.Storage.device)
            else:
                tmpb = b.Storage

            tmp = torch.tensordot(tmpa, tmpb, dims=(a_ind.tolist(), b_ind.tolist()))

            new_bonds = np.concatenate(
                [copy.deepcopy(a.bonds[aind_no_combine]), copy.deepcopy(b.bonds[bind_no_combine])])
            new_io = [(aind_no_combine[x] >= a.rowrank) for x in range(len(aind_no_combine))] + [
                (bind_no_combine[x] >= b.rowrank) for x in range(len(bind_no_combine))]
            new_labels = np.concatenate([copy.copy(a.labels[aind_no_combine]), copy.copy(b.labels[bind_no_combine])])

            new_io = np.array(new_io)
            # print(new_io)
            if len(new_bonds) > 0:
                mapper = np.argsort(new_io)
                new_bonds = new_bonds[mapper]
                new_labels = new_labels[mapper]
                tmp = tmp.permute(*mapper)

            out = UniTensor(bonds=new_bonds,
                            labels=new_labels,
                            rowrank=len(np.argwhere(new_io == 0)),
                            check=False)
            out._mac(torch_tensor=tmp)

            return out
        else:
            ## direct product

            if a.is_diag:
                tmpa = torch.diag(a.Storage)
            else:
                tmpa = a.Storage

            if b.is_diag:
                tmpb = torch.diag(b.Storage)
            else:
                tmpb = b.Storage

            tmp = torch.tensordot(tmpa, tmpb, dims=0)
            new_bonds = np.concatenate([copy.deepcopy(a.bonds), copy.deepcopy(b.bonds)])
            new_labels = np.concatenate([copy.copy(a.labels), copy.copy(b.labels)])
            new_io = [(x >= a.rowrank) for x in range(len(a.bonds))] + [(x >= b.rowrank) for x in range(len(b.bonds))]

            if len(new_bonds) > 0:
                mapper = np.argsort(new_io)
                new_bonds = new_bonds[mapper]
                new_labels = new_labels[mapper]
                tmp = tmp.permute(*mapper)

            out = UniTensor(bonds=new_bonds,
                            labels=new_labels,
                            rowrank=a.rowrank + b.rowrank,
                            check=False)
            out._mac(torch_tensor=tmp)
            return out
    else:
        raise Exception('Contract(a,b)', "[ERROR] a and b both have to be UniTensor")


## The functions that start with "_" are the private functions

def _CombineBonds(a, idxs, new_label, permute_back):
    """
    [Private function, should not be called directly by user]

    This function combines the bonds in input UniTensor [a] by the specified labels [label]. The bondType of the combined bonds will always follows the same bondType of bond in [a] with label of the largest index element in [label]

    Args:

        a:
            UniTensor

        idxs:

            index that to be combined. It should be a int list / numpy array of the label. All the bonds with specified labels in the current UniTensor  will be combined

        new_label:
            the new_label of the combined bond 

        permute_back:
            Set if the current bond should be permute back

            

    """
    if isinstance(a, UniTensor):

        idx_no_combine = np.setdiff1d(np.arange(len(a.labels)),
                                      idxs)  ## idx_no_combine will be from small to large, sorted!
        old_shape = np.array(a.shape)

        combined_dim = old_shape[idxs]
        combined_dim = np.prod(combined_dim)
        no_combine_dims = old_shape[idx_no_combine]

        ## Set new label if appears.
        if new_label is not None:
            newlbl = int(new_label)
            if newlbl in a.labels[idx_no_combine] or newlbl in a.labels[idxs[1:]]:
                raise Exception("_CombineBonds",
                                "[ERROR], cannot set new_label to %d as there will be duplicate bond with this label after combined" % newlbl)

            a.labels[idxs[0]] = newlbl

        ##------------------------------------
        ## master switch 
        if a.is_symm:
            ## symmetry
            # raise Exception("[Develope]")

            ## check if the combine are BRA or KET
            contype_inout = np.unique(a.braket[idxs])
            if len(contype_inout) != 1:
                raise Exception("_CombineBonds",
                                "[ERROR], label_to_combine should be all bra-bond or all ket-bond for Tensor with symmetry")

            if idxs[0] < a.rowrank:
                a.Permute(np.concatenate((idxs, idx_no_combine)), rowrank=len(idxs), by_label=False)
                ## put it on the contiguous form:
                a.Contiguous_()

                ## DEBUG >>>
                if DEBUG:
                    if not a.is_contiguous():
                        raise Exception("[ERROR][DEBUG][internal] non-contiguous!!")
                ## <<<

                ##[Fusion tree] >>>
                # new_Nin = a.rowrank
                for i in range(len(idxs) - 1):
                    # if idxs[1+i]<a.rowrank:
                    #    new_Nin-=1
                    a.bonds[0].combine(a.bonds[1 + i])
                ## <<<

                del_pos = np.arange(1, len(idxs), 1).astype(np.int)
                a.labels = np.delete(a.labels, del_pos)
                a.braket = np.delete(a.braket, del_pos)
                a.bonds = np.delete(a.bonds, del_pos)

                ##house keeping mappers 
                for b in range(len(a.Storage)):
                    a._Ket_invmapper_blks[b] = np.sum(a._Ket_invmapper_blks[b] * a._accu_off_in, axis=1)

                a._accu_off_in = np.array([1], dtype=np.int)
                a._mapper = np.arange(len(a.labels), dype=np.int)  ## contiguous, so we just init
                a._inv_mapper = copy.copy(a._mapper)

                a.rowrank = 1

            else:
                a.Permute(np.concatenate((idx_no_combine, idxs[::-1])), rowrank=len(a.labels) - len(idxs),
                          by_label=False)
                ## put it on the contiguous form:
                a.Contiguous_()
                print(a.labels)
                ## DEBUG >>>
                if DEBUG:
                    if not a.is_contiguous():
                        raise Exception("[ERROR][DEBUG][internal] non-contiguous!!")
                ## <<<

                ##[Fusion tree] >>>
                # new_Nin = a.rowrank
                for i in range(len(idxs) - 1):
                    # if idxs[1+i]<a.rowrank:
                    #    new_Nin-=1
                    a.bonds[-1].combine(a.bonds[-2 - i])
                ## <<<

                del_pos = np.arange(len(a.labels) - len(idxs), len(a.labels) - 1, 1).astype(np.int)
                print(del_pos)
                a.labels = np.delete(a.labels, del_pos)
                a.braket = np.delete(a.braket, del_pos)
                a.bonds = np.delete(a.bonds, del_pos)

                ##house keeping mappers 
                for b in range(len(a.Storage)):
                    a._Bra_invmapper_blks[b] = np.sum(a._Bra_invmapper_blks[b] * a._accu_off_out, axis=1)

                a._accu_off_out = np.array([1], dtype=np.int)
                a._mapper = np.arange(len(a.labels), dtype=np.int)  ## contiguous, so we just init
                a._inv_mapper = copy.copy(a._mapper)

                a.rowrank = len(a.labels) - 1

            if permute_back:
                a.braket_form()


        else:
            ## non-symm
            if a.is_diag:
                raise TypeError("_CombineBonds", "[ERROR] CombineBonds doesn't support diagonal matrix.")

            if a.braket is None:
                ## untagged type:

                if permute_back:

                    ##[Fusion tree] >>>
                    new_Nin = a.rowrank
                    for i in range(len(idxs) - 1):
                        if idxs[1 + i] < a.rowrank:
                            new_Nin -= 1
                        a.bonds[idxs[0]].combine(a.bonds[idxs[1 + i]])
                    ## <<<

                    mapper = np.concatenate([idxs, idx_no_combine])
                    a.Storage = a.Storage.permute(mapper.tolist()).contiguous().view(
                        np.append(combined_dim, no_combine_dims).tolist())

                    f_label = a.labels[idxs[0]]
                    a.bonds = np.delete(a.bonds, idxs[1:])
                    a.labels = np.delete(a.labels, idxs[1:])
                    
                    x = np.argwhere(a.labels == f_label)
                    final_mapper = np.insert(np.arange(1, len(a.bonds), 1).astype(np.int), x[0], 0)
                    a.Stoarge = a.Storage.permute(final_mapper.tolist())

                    a.rowrank = new_Nin

                else:
                    ##[Fusion tree] >>>
                    for i in range(len(idxs) - 1):
                        a.bonds[idxs[0]].combine(a.bonds[idxs[1 + i]])
                    ## <<<
                    if idxs[0] >= a.rowrank:
                        mapper = np.concatenate([idx_no_combine, idxs])
                        a.bonds = np.append(a.bonds[idx_no_combine], a.bonds[idxs[0]])
                        a.labels = np.append(a.labels[idx_no_combine], a.labels[idxs[0]])
                        a.Storage = a.Storage.permute(mapper.tolist()).contiguous().view(
                            np.append(no_combine_dims, combined_dim).tolist())
                        a.rowrank = len(a.labels) - 1
                    else:
                        mapper = np.concatenate([idxs, idx_no_combine])
                        a.bonds = np.append(a.bonds[idxs[0]], a.bonds[idx_no_combine])
                        a.labels = np.append(a.labels[idxs[0]], a.labels[idx_no_combine])
                        a.Storage = a.Storage.permute(mapper.tolist()).contiguous().view(
                            np.append(combined_dim, no_combine_dims).tolist())
                        a.rowrank = 1
            else:

                ## if the combine are BRA or KET
                contype_inout = np.unique(a.braket[idxs])
                if len(contype_inout) != 1:
                    raise Exception("_CombineBonds",
                                    "[ERROR], label_to_combine should be all bra-bond or all ket-bond for tagged-nonsymm Tensor")

                if permute_back:

                    ##[Fusion tree] >>>
                    new_Nin = a.rowrank
                    # print(a.bonds)
                    # print(a.bonds.shape)
                    for i in range(len(idxs) - 1):
                        if idxs[1 + i] < a.rowrank:
                            new_Nin -= 1
                        a.bonds[idxs[0]].combine(a.bonds[idxs[1 + i]])
                    ## <<<

                    mapper = np.concatenate([idxs, idx_no_combine])
                    a.Storage = a.Storage.permute(mapper.tolist()).contiguous().view(
                        np.append(combined_dim, no_combine_dims).tolist())

                    f_label = a.labels[idxs[0]]
                    a.bonds = np.delete(a.bonds, idxs[1:])
                    a.labels = np.delete(a.labels, idxs[1:])
                    a.braket = np.delete(a.braket, idxs[1:])

                    x = np.argwhere(a.labels == f_label)
                    final_mapper = np.insert(np.arange(1, len(a.bonds), 1).astype(np.int), x[0], 0)
                    a.Stoarge = a.Storage.permute(final_mapper.tolist())

                    a.rowrank = new_Nin
                else:

                    ##[Fusion tree] >>>
                    for i in range(len(idxs) - 1):
                        a.bonds[idxs[0]].combine(a.bonds[idxs[1 + i]])
                    ## <<<
                    
                    if idxs[0] >= a.rowrank:
                        mapper = np.concatenate([idx_no_combine, idxs])
                        a.bonds = np.append(a.bonds[idx_no_combine], a.bonds[idxs[0]])
                        a.labels = np.append(a.labels[idx_no_combine], a.labels[idxs[0]])
                        a.braket = np.append(a.braket[idx_no_combine], a.braket[idxs[0]])
                        a.Storage = a.Storage.permute(mapper.tolist()).contiguous().view(
                            np.append(no_combined_dims, combine_dim).tolist())
                        a.rowrank = len(a.labels) - 1
                       
                    else:
                        mapper = np.concatenate([idxs, idx_no_combine])
                        a.bonds = np.append(a.bonds[idxs[0]], a.bonds[idx_no_combine])
                        a.labels = np.append(a.labels[idxs[0]], a.labels[idx_no_combine])
                        a.braket = np.append(a.braket[idxs[0]], a.braket[idx_no_combine])
                        a.Storage = a.Storage.permute(mapper.tolist()).contiguous().view(
                            np.append(combined_dim, no_combine_dims).tolist())
                        a.rowrank = 1

    else:
        raise Exception("_CombineBonds(UniTensor,int_arr)", "[ERROR] )CombineBonds can only accept UniTensor")


def _Randomize(a):
    """
        @description: <private function> This function randomize a UniTensor.
        @params     :
                      a : UniTensor
        @return     : N/A

    """

    if isinstance(a, UniTensor):
        if a.is_symm:
            for s in range(len(a.Storage)):
                a.Storage[s] = torch.rand(a.Storage[s].shape, dtype=a.Storage[s].dtype, device=a.Storage[s].device)
        else:
            a.Storage = torch.rand(a.Storage.shape, dtype=a.Storage.dtype, device=a.Storage.device)


    else:
        raise Exception("_Randomize(UniTensor)", "[ERROR] _Randomize can only accept UniTensor")


def From_torch(torch_tensor, rowrank=None, labels=None, is_tag=False):
    """
    Construct UniTensor from torch.Tensor.

    If the input torch_tensor belongs to a autograd graph, the contructed UniTensor will preserve the role of the input torch_tensor in the computational graph.

    Args:
        torch_tensor:
            Torch.Tensor

        rowrank:
            int, The number of inbond. Note that the first [rowrank] bonds will be set to tor10.BD_IN, and the remaining bonds will be set to tor10.BD_OUT

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

        >>> y = tor10.From_torch(x,rowrank=1,labels=[4,5])
        >>> y.Print_diagram()
        -----------------------
        tensor Name : 
        tensor Rank : 2
        has_symmetry: False
        on device     : cpu
        is_diag       : False
                    -------------      
                   /             \     
             4 ____| 3         3 |____ 5  
                   \             /     
                    -------------  

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

        >>> y2 = tor10.From_torch(x2,rowrank=1)
        >>> print(y2.requires_grad())
        True


    """
    if not isinstance(torch_tensor, torch.Tensor):
        raise TypeError("From_torch", "[ERROR] can only accept torch.Tensor")

    shape = torch_tensor.shape

    if rowrank is not None:
        if rowrank > len(shape) or rowrank < 0:
            raise ValueError("From_torch", "[ERROR] rowrank exceed the rank of input torch tensor.")
    else:
        if len(shape) != 0:
            raise ValueError("From_torch", "[ERROR] rowrank must be set for a non rank-0 tensor")

    if labels is not None:
        if len(labels) != len(shape):
            raise TypeError("From_torch", "[ERROR] # of labels should match the rank of torch.Tensor")

    if is_tag:

        new_bonds = [Bond(shape[i], BD_KET) for i in range(rowrank)] + \
                    [Bond(shape[i], BD_BRA) for i in np.arange(rowrank, len(shape), 1)]
    else:
        new_bonds = [Bond(shape[i]) for i in range(len(shape))]

    if len(new_bonds) == 0:
        tmp = UniTensor(bonds=[], labels=[], rowrank=0, check=False)
        tmp._mac(torch_tensor=torch_tensor)
        return tmp
    else:
        tmp = UniTensor(bonds=new_bonds, labels=labels, rowrank=rowrank, check=False)
        tmp._mac(torch_tensor=torch_tensor)
        return tmp
