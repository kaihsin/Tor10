import torch, copy
import numpy as np
from . import Symmetry as Symm


#
# Find "DevNote" for the note attach on each functions that should be awared of for all the developer.
#
#


##Helper function for Bond:
def _fx_GetCommRows(A, B):
    # this is the internal function, to get the common row on two 2D numpy array
    # [Required for input]
    # 1. A and B should be 2D numpy array
    # 2. the number of col should be the same for A and B

    # Source: https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    dtype = {'names': ['f{}'.format(i) for i in range(A.shape[1])],
             'formats': A.shape[1] * [A.dtype]}

    C = np.intersect1d(A.view(dtype), B.view(dtype))

    # This last bit is optional if you're okay with "C" being a structured array...
    C = C.view(A.dtype).reshape(-1, A.shape[1])

    return C


##### Constants #######
class BD_KET:
    pass
class BD_BRA:
    pass

class BD_REG:
    pass


BondType = {BD_BRA:1,BD_KET:-1, BD_REG:0}
#BondType = {BD_KET:1,BD_BRA:-1,BD_REG:0}

## [For developer] Append this to extend the symmetry:

SymmetryTypes = {'U1': Symm.U1, 'Zn': Symm.Zn}


#######################


# noinspection PyStringFormat
class Bond:

    #
    # [0] bondType
    # [/] vector<Qnums> Qnums; ## This is multiple Q1
    # [x] vector<int> Qdegs;
    # [x] vector<int> offsets;
    # [x] bool withSymm

    ## [DevNote]:The qnums should be integer.

    def __init__(self, dim, bondType=BD_REG, qnums=None, sym_types=None):
        """
        Constructor of the Bond, it calls the member function Bond.assign().

        Args:

            dim:
                The dimension of the bond.
                It should be larger than 0 (>0)

            bondType:
                The type of the bond. It can be one of the following three types in current version. 
                    
                    1. BD_REG : regular bond 
                    2. BD_BRA : tag, indicating the basis space of the bond is a "bra" |v> 
                    3. BD_KET : tag, indicating the basis space of the bond os a "ket" <v| 
    

                [Note] For the bond with symmetry, it should always being tagged with either BD_BRA or BD_KET. A bond cannot be BD_REG with symmetry. 
    

            qnums:
                The quantum number(s) specify to the bond.
                The qnums should be a 2D numpy array or 2D list, with shape=(dim , No. of Symmetry). The No. of Symmetry can be arbitrary.
                
                [Note] the input qnums will be lexsort from large to small, and from first typs of symmetry to last typs of symmetry.

            sym_types:
                The Symmetry types specify to each Symmetry. if qnums is set, the default symmetry type is U1. 

        Example:

            Create an simple bond with dimension=4:

            >>> bd_r = tor10.Bond(3) # this is equivalent as "bd_r = tor10.Bond(3,tor10.BD_REG)"
            >>> print(bd_r)
            Dim = 3 |
            REG     :

            Create a bond with tag: BD_KET (ket-bond) with dimension=3:

            >>> bd_ket = tor10.Bond(3,tor10.BD_KET)
            >>> print(bd_ket)
            Dim = 3 |
            KET :

            Create an ket-bond of dimension=3 with single U1 symmetry, and quantum numbers=[-1,0,1] for each dimension:

            >>> bd_sym_U1 = tor10.Bond(3,tor10.BD_KET,qnums=[[-1],[0],[1]])
            >>> print(bd_sym_U1)
            Dim = 3 |
            KET     : U1::  +1 +0 -1

            The above example is equivalent to:

            >>> bd_sym_U1 = tor10.Bond(3,tor10.BD_KET,qnums=[[-1],[0],[1]],sym_types=[tor10.Symmetry.U1()])

            Create an symmetry bra-bond of dimension=3 with single Zn symmetry (n can be arbitrary positive Integer).

            1. bra-bond with Z2 symmetry, with quantum numbers=[0,1,0] for each dimension:

            >>> bd_sym_Z2 = tor10.Bond(3,tor10.BD_BRA,qnums=[[0],[1],[0]],sym_types=[tor10.Symmetry.Zn(2)])
            >>> print(bd_sym_Z2)
            Dim = 3 |
            BRA     : Z2::  +1 +0 +0

            2. ket-bond with Z4 symmetry, with quantum numbers=[0,2,3] for each dimension:

            >>> bd_sym_Z4 = tor10.Bond(3,tor10.BD_KET,qnums=[[0],[2],[3]],sym_types=[tor10.Symmetry.Zn(4)])
            >>> print(bd_sym_Z4)
            Dim = 3 |
            KET     : Z4::  +3 +2 +0

            Create a ket-bond of dimension=3 with multiple U1 symmetry (here we consider U1 x U1 x U1 x U1, so the No. of symmetry =4), with
            1st dimension quantum number = [-2,-1,0,-1],
            2nd dimension quantum number = [1 ,-4,0, 0],
            3rd dimension quantum number = [-8,-3,1, 5].
            ::
               bd_out_mulsym = tor10.Bond(3,tor10.BD_KET,qnums=[[-2,-1,0,-1],
                                                                [1 ,-4,0, 0],
                                                                [-8,-3,1, 5]])

            >>> print(bd_out_mulsym)
            Dim = 3 |
            KET     : U1::  +1 -2 -8
                      U1::  -4 -1 -3
                      U1::  +0 +0 +1
                      U1::  +0 -1 +5

            Create an symmetry bond of dimension=3 with U1 x Z2 x Z4 symmetry (here, U1 x Z2 x Z4, so the No. of symmetry = 3), with
            1st dimension quantum number = [-2,0,0],
            2nd dimension quantum number = [-1,1,3],
            3rd dimension quantum number = [ 1,0,2].
            ::
                bd_out_mulsym = tor10.Bond(3,tor10.BD_BRA,qnums=[[-2,0,0],
                                                                 [-1,1,3],
                                                                 [ 1,0,2]],
                                             sym_types=[tor10.Symmetry.U1(),
                                                        tor10.Symmetry.Zn(2),
                                                        tor10.Symmetry.Zn(4)])

            >>> print(bd_out_mulsym)
            Dim = 3 |
            BRA     : U1::  +1 -1 -2
                      Z2::  +0 +1 +0
                      Z4::  +2 +3 +0

        """
        # declare variable:
        self.bondType = None
        self.dim = None
        self.qnums = None
        self.nsym = 0
        self.sym_types = None

        # call :
        self.assign(dim, bondType, qnums, sym_types)

    def assign(self, dim, bondType=BD_REG, qnums=None, sym_types=None):
        """
        Assign a new property for the Bond.

        Args:

            dim:
                The dimension of the bond.
                It should be larger than 0 (>0)

            bondType:
                The type of the bond.
                It can be BD_BRA or BD_KET or BD_REG in current version. 

            qnums:
                The quantum number(s) specify to the bond.
                The qnums should be a 2D numpy array or 2D list, with shape=(dim , No. of Symmetry). The No. of Symmetry can be arbitrary.
            sym_types:
                The Symmetry types specify to each Symmetry. if qnums is set, the default symmetry type is U1.

        Example:

            For a ket-bond with dim=4, U1 x U1 x U1; there are 3 of U1 symmetry.
            The Bond can be initialize as:
            ::
                a = tor10.Bond(4,tor10.BD_BRA) # create instance
                a.assign(4,tor10.BD_KET,qnums=[[ 0, 1, 1],
                                               [-1, 2, 0],
                                               [ 0, 1,-1],
                                               [ 2, 0, 0]])

                                                 ^  ^  ^
                                                U1 U1 U1

            For a bra-bond with dim=3, U1 x Z2 x Z4; there are 3 symmetries.
            The Bond should be initialize as :
            ::
                b = tor10.Bond(3,tor10.BD_BRA)
                b.assign(3,tor10.BD_BRA,sym_types=[tor10.Symmetry.U1(),
                                                   tor10.Symmetry.Zn(2),
                                                   tor10.Symmetry.Zn(4)],
                                            qnums=[[-2, 0, 3],
                                                   [-1, 1, 1],
                                                   [ 2, 0, 0]])
                                                     ^  ^  ^
                                                    U1 Z2 Z4
        """

        # checking:

        if dim < 1:
            raise Exception("Bond.assign()", "[ERROR] Bond dimension must be greater than 0.")

        if not bondType in BondType:
            raise Exception("Bond.assign()", "[ERROR] bondType can only be BD_BRA, BD_KET or BD_REG")
            #raise Exception("Bond.assign()", "[ERROR] bondType can only be BD_BRA or BD_KET")

        if not qnums is None:
            if bondType is BD_REG:
                raise Exception("Bond.assign()","[ERROR] with qnums, bondType can only be BD_BRA or BD_KET")
            sp = np.shape(qnums)
            if len(sp) != 2:
                raise TypeError("Bond.assign()", "[ERROR] qnums must be a list of lists (2D list).")

            xdim = np.unique([len(qnums[x]) for x in range(len(qnums))]).flatten()
            if len(xdim) != 1:
                raise TypeError("Bond.assign()", "[ERROR] Number of symmetries must be the same for each dim.")

            if len(qnums) != dim:
                raise ValueError("Bond.assign()", "[ERROR] qnums must have the same elements as dim")
            self.nsym = xdim[0]
            self.qnums = np.array(qnums).astype(np.int)

            ## default is U1. this is to preserve the API
            if sym_types is None:
                self.sym_types = np.array([SymmetryTypes['U1']() for i in range(xdim[0])])
            else:
                if xdim[0] != len(sym_types):
                    raise TypeError("Bond.assign()", "[ERROR] Number of symmetry types must match qnums.")
                else:
                    ## checking :
                    for s in range(len(sym_types)):

                        # check the sym_types is a vaild symmetry class appears in SymmType dict.
                        if sym_types[s].__class__ not in SymmetryTypes.values():
                            raise TypeError("Bond.assign()", "[ERROR] invalid Symmetry Type.")

                        # check each qnum validity subject to the symmetry.
                        if not sym_types[s].CheckQnums(self.qnums[:, s]):
                            raise TypeError("Bond.assign()", "[ERROR] invalid qnum in Symmetry [%s] @ index: %d" % (
                                str(sym_types[s]), s))
                    self.sym_types = copy.deepcopy(sym_types)

            y = np.lexsort(self.qnums.T[::-1])[::-1]
            self.qnums = self.qnums[y,:]

        else:
            if sym_types is not None:
                raise ValueError("Bond.assign()", "[ERROR] the sym_type is assigned but no qnums is passed.")

        ## fill the members:
        self.bondType = bondType
        self.dim = dim

    # [DevNote]this is the dummy_change as uni10_2.0
    # [DevNote]This is the inplace change

    def change(self, new_bondType):
        """
        Change the type of the bond. 

        [Note] if the current bond has symmetry, it cannot be changed to BD_REG 

        Args:

            new_bondType: The new bond type to be changed. In current version, only BD_KET or BD_BRA or BD_REG. 

        """
        if self.bondType is not new_bondType:
            if not new_bondType in BondType:
                raise TypeError("Bond.change", "[ERROR] the bondtype can only be", BondType)
            if self.qnums is not None:
                if new_bondType == BD_REG:
                    raise TypeError("Bond.change","[ERROR] cannot change a bond with symmetry to BD_REG type")
            self.bondType = new_bondType

    # [DevNote] This is the inplace combine.

    def combine(self, bds, new_type=None):
        """
        Combine self with the bond that specified.

        Args:

            bds:
                the bond that is going to be combine with self.
                1. two bonds can only be combined when both have the same type. 
                2. two symmetry bonds can only be combined when both of the No. of symmetry are the same.

            new_type:
                the type of the new combined bond, it can only be BD_BRA, BD_KET or BD_REG in current version. If not specify, the bond Type will remains the same.

                [Note] if the combined bond has symmetry, it cannot be changed to BD_REG


        Example:
        ::
            a = tor10.Bond(3,tor10.BD_BRA)
            b = tor10.Bond(4,tor10.BD_KET)
            c = tor10.Bond(2,tor10.BD_BRA,qnums=[[0,1,-1],[1,1,0]])
            d = tor10.Bond(2,tor10.BD_KET,qnums=[[1,0,-1],[1,0,0]])
            e = tor10.Bond(2,tor10.BD_KET,qnums=[[1,0],[1,0]])

        Combine two non-symmetry bonds:
            >>> a.combine(b)
            >>> print(a)
            Dim = 12 |
            BRA      :

        Combine two symmetry bonds:
            >>> c.combine(d)
            >>> print(c)
            Dim = 4 |
            BRA     : U1::  +2 +2 +1 +1
                      U1::  +1 +1 +1 +1
                      U1::  +0 -1 -1 -2

        """
        ## if bds is Bond class
        if isinstance(bds, self.__class__):
            self.dim *= bds.dim
            if (self.qnums is None) != (bds.qnums is None):
                raise TypeError("Bond.combine", "[ERROR] Trying to combine symmetric and non-symmetric bonds.")
            if self.qnums is not None:
                # check number of symmetry.
                if self.nsym != len(bds.qnums[0]):
                    raise TypeError("Bond.combine",
                                    "[ERROR] Trying to combine bonds with different number of symmetries.")

                # check symmetry types
                for s in range(self.nsym):
                    if self.sym_types[s] != bds.sym_types[s]:
                        raise TypeError("Bond.combine", "[ERROR] Tryping to combine bonds with different symmetries.")

                # combine accroding to the rule:
                A = self.qnums.reshape(len(self.qnums), 1, self.nsym)
                B = bds.qnums.reshape(1, len(bds.qnums), self.nsym)

                self.qnums = []

                # [Dev Note] Using side effects of numpy.add to first reshape to
                # len(self.qnums)xlen(bds.qnums) arrays and add, will not work for non-Abelian symmetries

                for s in range(self.nsym):
                    self.qnums.append(self.sym_types[s].CombineRule(A[:, :, s], B[:, :, s]))

                self.qnums = np.array(self.qnums).reshape(self.nsym, -1).swapaxes(0, 1)

                # self.qnums = (self.qnums.reshape(len(self.qnums),1,self.nsym)+bds.qnums.reshape(1,len(bds.qnums),self.nsym)).reshape(-1,self.nsym)


        else:
            ## combine a list of bonds:
            for i in range(len(bds)):
                if not isinstance(bds[i], self.__class__):
                    raise TypeError("Bond.combine(bds)", "bds[%d] is not Bond class" % i)
                else:
                    self.dim *= bds[i].dim
                    if (self.qnums is None) != (bds[i].qnums is None):
                        raise TypeError("Bond.combine", "[ERROR] Trying to combine bonds with symm and non-symm")
                    if self.qnums is not None:
                        if self.nsym != len(bds[i].qnums[0]):
                            raise TypeError("Bond.combine",
                                            "[ERROR] Trying to combine bonds with different number of symm.")
                        for s in range(self.nsym):
                            if self.sym_types[s] != bds[i].sym_types[s]:
                                raise TypeError("Bond.combine",
                                                "[ERROR] Tryping to combine bonds with different symmetries.")

                        ## combine accroding to the rule:
                        A = self.qnums.reshape(len(self.qnums), 1, self.nsym)
                        B = bds[i].qnums.reshape(1, len(bds[i].qnums), self.nsym)

                        self.qnums = []
                        for s in range(self.nsym):
                            # [Dev Note] Using side effects of numpy.add to first reshape to
                            # len(self.qnums)xlen(bds.qnums) array and add, will not work for non-Abelian symmetries

                            self.qnums.append(self.sym_types[s].CombineRule(A[:, :, s], B[:, :, s]))

                        self.qnums = np.array(self.qnums).reshape(self.nsym, -1).swapaxes(0, 1)

                        # self.qnums = (self.qnums.reshape(len(self.qnums),1,self.nsym)+bds[i].qnums.reshape(1,len(bds[i].qnums),self.nsym)).reshape(-1,self.nsym)

        ## checking change type
        if not new_type is None:
            if not new_type in BondType:
                raise Exception("Bond.combine(bds,new_type)", "[ERROR] new_type can only be", BondType)
            else:
                self.change(new_type)

    def GetUniqueQnums(self,return_degeneracy=False):
        """
            Return a (sorted) Unique Qnums by remove the duplicates. It can only be call on a bond with symmetry.

            Args:
                
                return_degeneracy [default: False]: 
                    return the defeneracy of each qnums:
                
            return:
                2D numpy.array with shape (# of unique qnum-set, # of symmetry)

        """
        if self.qnums is None:
            raise TypeError("Bond.GetUniqueQnums", "[ERROR] cannot get qnums from a non-sym bond.")

        if return_degeneracy:
            uqn = np.unique(self.qnums, axis=0)
            deg = []
            for q in uqn:
                deg.append( len(np.argwhere((self.qnums == uqn).all(axis=1)).flatten()))
            deg = np.array(deg)
            return uqn,deg
        else:
            return np.unique(self.qnums, axis=0)

    def GetDegeneracy(self, *qnums):
        """
            Return degenracy of a specify quantum number set. If can only be call on a bond with symmetry.
            
            Args:

                *qnums: 
                    The quantum number set

            Return:

                int, degeneracy of the quantum number set
        """
        if self.qnums is None:
            raise TypeError("Bond.GetDegeneracy","[ERROR] cannot get degenerate from a non-sym bond.")

        deg = len(np.argwhere((self.qnums == np.array(qnums).astype(np.int)).all(axis=1)).flatten())
        return deg


    def __mul__(self,val):
        """ 
            This is use to reverse the quantum numbers ( for calculate unmatched bra/ket bond blocks )
        """
        if (val != -1) and (val != 1):
            raise Exception("[ERROR] val can only be +1 or -1")
       
        self.qnums *= val
        return self

    ## Print layout
    def __print(self):
        print("Dim = %d |" % self.dim, end="\n")
        if self.bondType is BD_REG:
            print("REG     :", end='')
        elif self.bondType is BD_BRA:
            print("BRA     :", end='')
        elif self.bondType is BD_KET:
            print("KET     :", end='')
        else:
            raise Exception("[Internal error][Invalid bondType of current Bond.]")

        if self.qnums is not None:
            for n in range(self.nsym):
                print(" %s:: " % (str(self.sym_types[n])), end='')
                for idim in range(len(self.qnums)):
                    print(" %+d" % (self.qnums[idim, n]), end='')
                print("\n         ", end='')
        else:
            print("\n", end="")

    def __str__(self):
        self.__print()
        return ""

    def __repr__(self):
        self.__print()
        return ""

    ## Arithmic:
    def __eq__(self, rhs):
        """
        Compare two bonds. Return True if [dim], [bondType] and [qnums] are all the same.

        example:
        ::
            bd_x = tor10.Bond(3,tor10.BD_BRA)
            bd_y = tor10.Bond(4,tor10.BD_BRA)
            bd_z = tor10.Bond(3,tor10.BD_BRA)

        >>> print(bd_x==bd_z)
        True

        >>> print(bd_x is bd_z)
        False

        >>> print(bd_x==bd_y)
        False

        """
        if isinstance(rhs, self.__class__):
            iSame = (self.dim == rhs.dim) and (self.bondType == rhs.bondType)
            if not iSame:
               return False

            if self.qnums is None:
                if rhs.qnums is not None:
                    return False
            else:
                if rhs.qnums is None:
                    return False

                iSame = iSame and (self.qnums == rhs.qnums).all()
                for s in range(self.nsym):
                    iSame = iSame and (self.sym_types[s] == rhs.sym_types[s])

            return iSame
        else:
            raise ValueError("Bond.__eq__", "[ERROR] invalid comparison between Bond object and other type class.")
