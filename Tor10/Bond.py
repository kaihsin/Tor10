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
    # [x] vector<Qnums> Qnums;
    # [x] vector<int> Qdegs;
    # [x] vector<int> offsets;
    # [x] bool withSymm

    def __init__(self, bondType, dim):
        #declare variable:
        self.bondType = None
        self.dim      = None

        #call :
        self.assign(bondType,dim)
 
    def assign(self,bondType, dim):
        #checking:
        if dim < 1: 
            raise Exception("Bond.assign()","[ERROR] Bond dimension must > 0") 
        if not bondType is BD_IN and not bondType is BD_OUT:
            raise Exception("Bond.assign()","[ERROR] bondType can only be BD_IN or BD_OUT")       

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
    def combine(bds,new_type=None):
        
        ## if bds is Bond class 
        if isinstance(bds,self.__class__):
            self.dim *= bds.dim
        else:
            for i in range(len(bds)):
                if not isinstance(bds[i],self.__class__):
                    raise TypeError("Bond.combine(bds)","bds[%d] is not Bond class"%(i))
                else:
                   self.dim *= bds[i].dim
        
        ## checking change type
        if not new_type is None:
            if not new_type is BD_IN and not new_type is BD_OUT:
                raise Exception("Bond.combine(bds,new_type)","[ERROR] new_type can only be BD_IN or BD_OUT")       
            else:
                self.change(new_type)


    ## Print layout
    def __print(self):

        if(self.bondType is BD_IN):
            print("IN : ")
        else:
            print("OUT : ")

        print("Dim = %d"%(self.dim))

    def __str__(self):
        self.__print()    
        return ""
    
    def __repr__(self):
        self.__print()
        return ""

    ## Arithmic:
    def __eq__(self,rhs):
        if isinstance(rhs,self.__class__):
            return (self.dim == rhs.dim) and (self.bondType == rhs.bondType)
        else:
            raise ValueError("Bond.__eq__","[ERROR] invalid comparison between Bond object and other type class.")


