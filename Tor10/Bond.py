import torch, copy
import numpy as np
from . import Symmetry as Symm


#
# Find "DevNote" for the note attach on each functions that should be awared of for all the developer.
#
#


##### Constants #######
#class BD_INWARD:
#    pass
#class BD_OUTWARD:
#    pass
class BD_REGULAR:
    pass

#BndType = [BD_INWARD,BD_OUTWARD,BD_REGULAR]
BndType = [BD_REGULAR]


## [For developer] Append this to extend the symmetry:
SymmType = {'U1':Symm.U1,'Zn':Symm.Zn}

#######################


class Bond():
    
    #
    # [0] bondType
    # [/] vector<Qnums> Qnums; ## This is multiple Q1
    # [x] vector<int> Qdegs;   
    # [x] vector<int> offsets;
    # [x] bool withSymm

    ## [DevNote]:The qnums should be integer.

    def __init__(self, dim, bondType=BD_REGULAR,qnums=None,sym_types=None):
        """
        Constructor of the Bond, it calls the member function Bond.assign().

        Args:
            
            dim:
                The dimension of the bond. 
                It should be larger than 0 (>0)
            bondType:
                The type of the bond. 
                It can be BD_REGULAR in current version. [BD_INWARD , BD_OUTWARD for brading is under developing.]
                default [BD_REGULAR] 
            qnums:
                The quantum number(s) specify to the bond. 
                The qnums should be a 2D numpy array or 2D list, with shape=(dim , No. of Symmetry). The No. of Symmetry can be arbitrary. 
            sym_types:
                The Symmetry types specify to each Symmetry. if qnums is set, the default symmetry type is U1. 

        Example:

            Create an non-symmetry bond with dimension=3:
            ::
                bd_in = Tor10.Bond(3,Tor10.BD_REGULAR)
            
            The above example is equivalent to:
            ::
                bd_in = Tor10.Bond(3)
            
    
            Create an symmetry bond of dimension=3 with single U1 symmetry, and quantum numbers=[-1,0,1] for each dimension:
            ::
                bd_out_sym = Tor10.Bond(3,qnums=[[-1],[0],[1]])
    
            Create an symmetry bond of dimension=3 with multiple U1 symmetry (here we consider U1 x U1 x U1 x U1, so the No. of symmetry =4), with 
            1st dimension quantum number = [-1,-1,0,-1],
            2nd dimension quantum number = [1 ,-1,0, 0],
            3rd dimension quantum number = [0 , 0,1, 0].
            ::
                bd_out_mulsym = Tor10.Bond(3,qnums=[[-1,-1,0,-1],
                                                    [1 ,-1,0, 0],
                                                    [0 , 0,1, 0]])
 
            Create an symmetry bond of dimension=3 with U1 x Z2 x Z4 symmetry (here, U1 x Z2 x Z4, so the No. of symmetry = 3), with
            1st dimension quantum number = [-2,0,0],
            2nd dimension quantum number = [-1,1,3],
            3rd dimension quantum number = [ 1,0,2].
            ::
                bd_out_mulsym = Tor10.Bond(3,qnums=[[-2,0,0],
                                                    [-1,1,3],
                                                    [ 1,0,2]],
                                             sym_types=[Tor10.Symmetry.U1(),
                                                        Tor10.Symmetry.Zn(2),
                                                        Tor10.Symmetry.Zn(4)]) 
         

        """
        #declare variable:
        self.bondType = None
        self.dim      = None
        self.qnums    = None
        self.nsym     = 0
        self.sym_types = None

        #call :
        self.assign(dim,bondType,qnums,sym_types)
 
    def assign(self,dim, bondType = BD_REGULAR, qnums=None,sym_types=None):
        """
        Assign a new property for the Bond.

        Args:

            dim:
                The dimension of the bond. 
                It should be larger than 0 (>0)
            bondType:
                The type of the bond. 
                It can be BD_REGULAR in current version. [BD_INWARD , BD_OUTWARD for brading is under developing.]
                default [BD_REGULAR] 
            qnums:
                The quantum number(s) specify to the bond. 
                The qnums should be a 2D numpy array or 2D list, with shape=(dim , No. of Symmetry). The No. of Symmetry can be arbitrary. 
            sym_types:
                The Symmetry types specify to each Symmetry. if qnums is set, the default symmetry type is U1. 

        Example:

            For a bond with dim=4, U1 x U1 x U1; there are 3 of U1 symmetry.
            The Bond can be initialize as:
            ::
                a = Tor10.Bond(4) # create instance
                a.assign(4 ,qnums=[[ 0, 1, 1],
                                   [-1, 2, 0],
                                   [ 0, 1,-1],
                                   [ 2, 0, 0]])
             
                                     ^  ^  ^
                                    U1 U1 U1

            For a bond with dim=3, U1 x Z2 x Z4; there are 3 symmetries.
            The Bond should be initialize as :
            ::
                b = Tor10.Bond(3)
                b.assign(3,sym_types=[Tor10.Symmetry.U1(),
                                      Tor10.Symmetry.Zn(2),
                                      Tor10.Symmetry.Zn(4)],
                                            qnums=[[-2, 0, 3],
                                                   [-1, 1, 1],
                                                   [ 2, 0, 0]])
                                                     ^  ^  ^
                                                    U1 Z2 Z4
        """

        #checking:
        if dim < 1: 
            raise Exception("Bond.assign()","[ERROR] Bond dimension must > 0") 

        if not bondType in BndType:
            raise Exception("Bond.assign()","[ERROR] bondType can only be BD_INWARD , BD_OUTWARD or BD_REGULAR")       

        if not qnums is None:
            sp = np.shape(qnums)
            if len(sp) != 2:
                raise TypeError("Bond.assign()","[ERROR] qnums must be list of list (2D list).")

            xdim = np.unique([len(qnums[x]) for x in range(len(qnums))]).flatten()
            if len(xdim) != 1:
                raise TypeError("Bond.assign()","[ERROR] the number of multiple symm must be the same for each dim.")

            if len(qnums) != dim:
                raise ValueError("Bond.assign()","[ERROR] qnums must have the same elements as the dim")        
            self.nsym = xdim[0]
            self.qnums = np.array(qnums).astype(np.int)

            ## default is U1. this is to preserve the API
            if sym_types is None:
                self.sym_types = np.array([SymmType['U1']() for i in range(xdim[0])])
            else:
                if xdim[0] != len(sym_types):
                    raise TypeError("Bond.assign()","[ERROR] the number of multiple symm types must match with qnums")
                else:
                    ## checking :
                    for s in range(len(sym_types)):

                        # check the sym_types is a vaild symmetry class appears in SymmType dict.
                        if sym_types[s].__class__ not in SymmType.values():
                            raise TypeError("Bond.assign()","[ERROR] invalid Symmetry Type.")

                        # check each qnum validity subject to the symmetry.
                        if not sym_types[s].CheckQnums(self.qnums[:,s]):
                            raise TypeError("Bond.assign()","[ERROR] invalid qnum in Symmetry [%s] @ index: %d"%(str(sym_types[s]),s))
                    self.sym_types = copy.deepcopy(sym_types)



        else:
            if sym_types is not None:
                raise ValueError("Bond.assign()","[ERROR] the sym_type is assigned but no qnums is passed.")

       ## fill the members:
        self.bondType = bondType
        self.dim      = dim

    
    #[DevNote]this is the dummy_change as uni10_2.0
    #[DevNote]This is the inplace change
    def change(self,new_bondType):
        """ 
        Change the type of the bond

        Args:

            new_bondType: The new bond type to be changed. In current version, only 

        """
        if(self.bondType is not new_bondType):
            if not new_bondType in BndType:
                raise TypeError("Bond.change","[ERROR] the bondtype can only be",BndType)
            self.bondType = new_bondType


    #[DevNote] This is the inplace combine.
    def combine(self,bds,new_type=None):
        """ 
        Combine self with the bond that specified.

        Args:

            bds: 
                the bond that is going to be combine with self.
                1. A non-symmetry bond cannot combine with a symmetry bond, and vice versa.
                2. two symmetry bonds can only be combined when both of the No. of symmetry are the same.

            new_type:
                the type of the new combined bond, it can only be BD_REGULAR in current version. [BD_INWARD , BD_OUTWARD is currently under developing for brading/fermionic system] If not specify, the bond Type will remains the same.

        Example:
        ::
            a = Tor10.Bond(3)
            b = Tor10.Bond(4)
            c = Tor10.Bond(2,qnums=[[0,1,-1],[1,1,0]])
            d = Tor10.Bond(Tor10.BD_REGULAR,2,qnums=[[1,0,-1],[1,0,0]]) 
            e = Tor10.Bond(Tor10.BD_REGULAR,2,qnums=[[1,0],[1,0]])

        Combine two non-symmetry bonds:
            >>> a.combine(b)
            >>> print(a)
            Dim = 12 |
            IN  :
            
        Combine two symmetry bonds:
            >>> c.combine(d)
            >>> print(c)
            Dim = 4 |
            OUT :U1::  +1 +1 +2 +2
                 U1::  +1 +1 +1 +1
                 U1::  -2 -1 -1 +0

        """
        ## if bds is Bond class 
        if isinstance(bds,self.__class__):
            self.dim *= bds.dim
            if (self.qnums is None) != (bds.qnums is None):
                raise TypeError("Bond.combine","[ERROR] Trying to combine bonds with symm and non-symm")
            if self.qnums is not None:
                # check number of symmetry.
                if self.nsym != len(bds.qnums[0]):
                    raise TypeError("Bond.combine","[ERROR] Trying to combine bonds with different number of symm.")
                
                # check symmetry types
                for s in range(self.nsym):
                    if self.sym_types[s] != bds.sym_types[s]:
                        raise TypeError("Bond.combine","[ERROR] Tryping to combine bonds with different symmetries.")


                ## combine accroding to the rule:
                A = self.qnums.reshape(len(self.qnums),1,self.nsym)
                B = bds.qnums.reshape(1,len(bds.qnums),self.nsym)
                
                self.qnums = []
                for s in range(self.nsym):
                    self.qnums.append(self.sym_types[s].CombineRule(A[:,:,s],B[:,:,s]))
                
                self.qnums = np.array(self.qnums).reshape(self.nsym,-1).swapaxes(0,1)
                
                #self.qnums = (self.qnums.reshape(len(self.qnums),1,self.nsym)+bds.qnums.reshape(1,len(bds.qnums),self.nsym)).reshape(-1,self.nsym)

                
        else:
            ## combine a list of bonds:
            for i in range(len(bds)):
                if not isinstance(bds[i],self.__class__):
                    raise TypeError("Bond.combine(bds)","bds[%d] is not Bond class"%(i))
                else:
                    self.dim *= bds[i].dim
                    if (self.qnums is None) != (bds[i].qnums is None):
                        raise TypeError("Bond.combine","[ERROR] Trying to combine bonds with symm and non-symm")
                    if self.qnums is not None: 
                        if self.nsym != len(bds[i].qnums[0]):
                            raise TypeError("Bond.combine","[ERROR] Trying to combine bonds with different number of symm.")
                        for s in range(self.nsym):
                            if self.sym_types[s] != bds[i].sym_types[s]:
                                raise TypeError("Bond.combine","[ERROR] Tryping to combine bonds with different symmetries.")

                        ## combine accroding to the rule:
                        A = self.qnums.reshape(len(self.qnums),1,self.nsym)
                        B = bds[i].qnums.reshape(1,len(bds[i].qnums),self.nsym)
                        
                        self.qnums = []
                        for s in range(self.nsym):
                            self.qnums.append(self.sym_types[s].CombineRule(A[:,:,s],B[:,:,s]))
                        
                        self.qnums = np.array(self.qnums).reshape(self.nsym,-1).swapaxes(0,1)
                        
                        #self.qnums = (self.qnums.reshape(len(self.qnums),1,self.nsym)+bds[i].qnums.reshape(1,len(bds[i].qnums),self.nsym)).reshape(-1,self.nsym)


        ## checking change type
        if not new_type is None:
            if not new_type in BndType:
                raise Exception("Bond.combine(bds,new_type)","[ERROR] new_type can only be",BndType)       
            else:
                self.change(new_type)


    ## Print layout
    def __print(self):
        print("Dim = %d |"%(self.dim),end="\n")
        """
        if(self.bondType is BD_INWARD):
            print("INWARD  :",end='')
            if not self.qnums is None:
                for n in range(self.nsym):
                    print("%s:: "%(str(self.sym_types[n])),end='')
                    for idim in range(len(self.qnums)):
                         print(" %+d"%(self.qnums[idim,n]),end='')
                    print("\n     ",end='')
            print("\n",end="")
        elif(self.bondType is BD_OUTWARD):
            print("OUTWARD :",end='')
            if not self.qnums is None:
                for n in range(self.nsym):
                    print("%s:: "%(str(self.sym_types[n])),end='')
                    for idim in range(len(self.qnums)):
                         print(" %+d"%(self.qnums[idim,n]),end='')
                    print("\n     ",end='')
            print("\n",end="")
        
        else:
        """
        if(self.bondType is BD_REGULAR):
            print("REGULAR :",end='')
            if not self.qnums is None:
                for n in range(self.nsym):
                    print(" %s:: "%(str(self.sym_types[n])),end='')
                    for idim in range(len(self.qnums)):
                         print(" %+d"%(self.qnums[idim,n]),end='')
                    print("\n         ",end='')
            print("\n",end="")

    def __str__(self):
        self.__print()    
        return ""
    
    def __repr__(self):
        self.__print()
        return ""

    ## Arithmic:
    def __eq__(self,rhs):
        if isinstance(rhs,self.__class__):
            iSame = (self.dim == rhs.dim) and (self.bondType == rhs.bondType) and (self.qnums == rhs.qnums)            
            if not self.qnums is None:
                for s in range(self.nsym):
                    iSame = iSame and (self.sym_types[s] == rhs.sym_types[s])

            return iSame
        else:
            raise ValueError("Bond.__eq__","[ERROR] invalid comparison between Bond object and other type class.")


