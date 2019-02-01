import torch, copy
import numpy as np


#
# Find "DevNote" for the note attach on each functions that should be awared of for all the developer.
#
#


##### Constants #######
class BD_IN:
    pass
class BD_OUT:
    pass

#######################


class Bond():
    
    #
    # [0] bondType
    # [/] vector<Qnums> Qnums; ## This is multiple Q1
    # [x] vector<int> Qdegs;   
    # [x] vector<int> offsets;
    # [x] bool withSymm

    ## [DevNote]:The qnums should be integer.

    def __init__(self, bondType, dim,qnums=None):
        """
        Constructor of the Bond, it calls the member function Bond.assign().

        Args:

            bondType:
                The type of the bond. 
                It can only be either BD_IN or BD_OUT 
            dim:
                The dimension of the bond. 
                It should be larger than 0 (>0)
            qnums:
                The quantum number(s) specify to the bond. 
                The qnums should be a 2D numpy array or 2D list, with shape=(dim , No. of Symmetry). The No. of Symmetry can be arbitrary. 

        Example:

            Create an non-symmetry in-bond with dimension=3:
            ::
                bd_in = Tor10.Bond(Tor10.BD_IN, 3)
    
            Create an symmetry out-bond of dimension=3 with single U1 symmetry, and quantum numbers=[-1,0,1] for each dimension:
            ::
                bd_out_sym = Tor10.Bond(Tor10.BD_OUT,3,qnums=[[-1],[0],[1]])
    
            Create an symmetry out-bond of dimension=3 with multiple U1 symmetry (here we consider U1 x U1 x U1 x U1, so the No. of symmetry =4), with 
            1st dimension quantum number = [-1,-1,0,-1],
            2nd dimension quantum number = [1 ,-1,0, 0],
            3rd dimension quantum number = [0 , 0,1, 0].
            ::
                bd_out_mulsym = Tor10.Bond(Tor10.BD_OUT,3,qnums=[[-1,-1,0,-1],
                                                                 [1 ,-1,0, 0],
                                                                 [0 , 0,1, 0]]) 
        """
        #declare variable:
        self.bondType = None
        self.dim      = None
        self.qnums    = None
        self.nsym     = 0

        #call :
        self.assign(bondType,dim,qnums)
 
    def assign(self,bondType, dim,qnums=None):
        """
        Assign a new property for the Bond.

        Args:

            bondType:
                The type of the bond. 
                It can only be either BD_IN or BD_OUT 
            dim:
                The dimension of the bond. 
                It should be larger than 0 (>0)
            qnums:
                The quantum number(s) specify to the bond. 
                The qnums should be a 2D numpy array or 2D list, with shape=(dim , No. of Symmetry). The No. of Symmetry can be arbitrary. 
        
        Example:

            For a in-bond with dim=4, U1 x U1 x Z2; there are 3 types of symmetry.
            The Bond should be initialize as:
            ::
                a = Tor10.Bond(Tor10.BD_OUT,4) # create instance
                a.assign(BD_IN,4,qnums=[[ 0, 1, 1],
                                        [-1, 2, 0],
                                        [ 0, 1,-1],
                                        [ 2, 0, 0]])
             
                                          ^  ^  ^
                                         U1 U1 Z2

        """

        #checking:
        if dim < 1: 
            raise Exception("Bond.assign()","[ERROR] Bond dimension must > 0") 

        if not bondType is BD_IN and not bondType is BD_OUT:
            raise Exception("Bond.assign()","[ERROR] bondType can only be BD_IN or BD_OUT")       

        if not qnums is None:
            sp = np.shape(qnums)
            if len(sp) != 2:
                raise TypeError("Bond.assign()","[ERROR] qnums must be list of list.")
            xdim = np.unique([len(qnums[x]) for x in range(len(qnums))]).flatten()
            if len(xdim) != 1:
                raise TypeError("Bond.assign()","[ERROR] the number of multiple symm must be the same for each dim.")
            if len(qnums) != dim:
                raise ValueError("Bond.assign()","[ERROR] qnums must have the same elements as the dim")        
            self.nsym = xdim[0]
            self.qnums = np.array(qnums).astype(np.int)
 
       ## fill the members:
        self.bondType = bondType
        self.dim      = dim

    
    #[DevNote]this is the dummy_change as uni10_2.0
    #[DevNote]This is the inplace change
    def change(self,new_bondType):
        """ 
        Change the type of the bond

        Args:

            new_bondType: The new bond type to be changed.

        """
        if(self.bondType is not new_bondType):
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
                the type of the new combined bond, it can only be either BD_IN or BD_OUT. If not specify, the bond Type will remains the same.

        Example:
        ::
            a = Tor10.Bond(BD_IN,3)
            b = Tor10.Bond(BD_OUT,4)
            c = Tor10.Bond(BD_OUT,2,qnums=[[0,1,-1],[1,1,0]])
            d = Tor10.Bond(BD_OUT,2,qnums=[[1,0,-1],[1,0,0]]) 
            e = Tor10.Bond(BD_OUT,2,qnums=[[1,0],[1,0]])

        Combine two non-symmetry bonds:
            >>> a.combine(b)
            >>> print(a)
            Dim = 12 |
            IN  :
            
        Combine two symmetry bonds:
            >>> c.combine(d)
            >>> print(c)
            Dim = 4 |
            OUT : +1 +1 +2 +2
                  +1 +1 +1 +1
                  -2 -1 -1 +0
        """
        ## if bds is Bond class 
        if isinstance(bds,self.__class__):
            self.dim *= bds.dim
            if (self.qnums is None) != (bds.qnums is None):
                raise TypeError("Bond.combine","[ERROR] Trying to combine bonds with symm and non-symm")
            if self.qnums is not None:
                if self.nsym != len(bds.qnums[0]):
                    raise TypeError("Bond.combine","[ERROR] Trying to combine bonds with different number of type of symm.")
                self.qnums = (self.qnums.reshape(len(self.qnums),1,self.nsym)+bds.qnums.reshape(1,len(bds.qnums),self.nsym)).reshape(-1,self.nsym)

                
        else:
            
            for i in range(len(bds)):
                if not isinstance(bds[i],self.__class__):
                    raise TypeError("Bond.combine(bds)","bds[%d] is not Bond class"%(i))
                else:
                    self.dim *= bds[i].dim
                    if (self.qnums is None) != (bds[i].qnums is None):
                        raise TypeError("Bond.combine","[ERROR] Trying to combine bonds with symm and non-symm")
                    if self.qnums is not None: 
                        if self.nsym != len(bds[i].qnums[0]):
                            raise TypeError("Bond.combine","[ERROR] Trying to combine bonds with different number of type of symm.")
                        self.qnums = (self.qnums.reshape(len(self.qnums),1,self.nsym)+bds[i].qnums.reshape(1,len(bds[i].qnums),self.nsym)).reshape(-1,self.nsym)


        ## checking change type
        if not new_type is None:
            if not new_type is BD_IN and not new_type is BD_OUT:
                raise Exception("Bond.combine(bds,new_type)","[ERROR] new_type can only be BD_IN or BD_OUT")       
            else:
                self.change(new_type)


    ## Print layout
    def __print(self):
        print("Dim = %d |"%(self.dim),end="\n")

        if(self.bondType is BD_IN):
            print("IN  :",end='')
            if not self.qnums is None:
                for n in range(self.nsym):
                    for idim in range(len(self.qnums)):
                         print(" %+d"%(self.qnums[idim,n]),end='')
                    print("\n     ",end='')
            print("\n",end="")
        else:
            print("OUT :",end='')
            if not self.qnums is None:
                for n in range(self.nsym):
                    for idim in range(len(self.qnums)):
                         print(" %+d"%(self.qnums[idim,n]),end='')
                    print("\n     ",end='')
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
            return (self.dim == rhs.dim) and (self.bondType == rhs.bondType) and (self.qnums == rhs.qnums)
        else:
            raise ValueError("Bond.__eq__","[ERROR] invalid comparison between Bond object and other type class.")


