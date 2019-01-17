import torch, copy
import numpy as np


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

    ## Develop note:
    ## khw: 1. The qnums should be integer.

    def __init__(self, bondType, dim,qnums=None):
        #declare variable:
        self.bondType = None
        self.dim      = None
        self.qnums    = None

        #call :
        self.assign(bondType,dim,qnums)
 
    def assign(self,bondType, dim,qnums=None):
        """
            @description:
            @params: 
                    bondType: The bond type, can be either BD_IN or BD_OUT
                    dim     : The dimension of the bond.
                    qnums   : an array, with shape (# of dim, # of symmetry type)
            @example:
                    for a in-bond with dim=4, U1 x U1 x Z2; there are 3 types of symmetry.
                    The Bond should be initialize as :
                    assign(BD_IN,3,qnums=[[ 0, 1, 1],\
                                          [-1, 2, 0],\
                                          [ 0, 1,-1],\
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
            if len(xdim) > 1:
                raise TypeError("Bond.assign()","[ERROR] the number of multiple symm must be the same for each dim.")
            if len(qnums) != dim:
                raise ValueError("Bond.assign()","[ERROR] qnums must have the same elements as the dim")        
            
            self.qnums    = np.array(qnums).astype(np.int)
 
       ## fill the members:
        self.bondType = bondType
        self.dim      = dim

    """ 
       this is the dummy_change as uni10_2.0
       Since there is no Qnum & Symm right now, so we only need dummy_change
       This is the inplace change
    """
    def change(self,new_bondType):
        if(self.bondType is not new_bondType):
            self.bondType = new_bondType

    """
        This is the inplace combine without Qnum & Symm.
    """
    def combine(self,bds,new_type=None):
        ## if bds is Bond class 
        if isinstance(bds,self.__class__):
            self.dim *= bds.dim
            nsym = None
            if self.qnums is not None:
                nsym = len(self.qnums[0])
            if (nsym is None) != (bds.qnums is None):
                raise TypeError("Bond.combine","[ERROR] Trying to combine bonds with symm and non-symm")
            if nsym is not None:
                if nsym != len(bds.qnums[0]):
                    raise TypeError("Bond.combine","[ERROR] Trying to combine bonds with different number of type of symm.")
                self.qnums = (self.qnums.reshape(len(self.qnums),1,nsym)+bds.qnums.reshape(1,len(bds.qnums),nsym)).reshape(-1,nsym)

                
        else:
            nsym = None
            if not self.qnums is None:
                nsym = len(self.qnums[0])
            
            for i in range(len(bds)):
                if not isinstance(bds[i],self.__class__):
                    raise TypeError("Bond.combine(bds)","bds[%d] is not Bond class"%(i))
                else:
                    self.dim *= bds[i].dim
                    if (nsym is None) != (bds[i].qnums is None):
                        raise TypeError("Bond.combine","[ERROR] Trying to combine bonds with symm and non-symm")
                    if nsym is not None: 
                        if nsym != len(bds[i].qnums[0]):
                            raise TypeError("Bond.combine","[ERROR] Trying to combine bonds with different number of type of symm.")
                        self.qnums = (self.qnums.reshape(len(self.qnums),1,nsym)+bds[i].qnums.reshape(1,len(bds[i]),nsym)).reshape(-1,nsym)


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
            print("IN :",end='')
            if not self.qnums is None:
                for nsym in range(len(self.qnums[0])):
                    for idim in range(len(self.qnums)):
                         print(" %+d"%(self.qnums[idim,nsym]),end='')
                    print("\n    ",end='')
            print("\n",end="")
        else:
            print("OUT :",end='')
            if not self.qnums is None:
                for nsym in range(len(self.qnums[0])):
                    for idim in range(len(self.qnums)):
                         print(" %+d"%(self.qnums[idim,nsym]),end='')
                    print("\n    ",end='')
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


