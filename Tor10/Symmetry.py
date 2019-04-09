import numpy as np 
import torch as tor

class U1:
    def __init__(self):
        pass 

    def CombineRule(self,A,B):
        return (A+B)

    def __repr__(self):
        return 'U1'

    def __str__(self):
        return 'U1'

    def __eq__(self,rhs):
        
        return self.__class__ == rhs.__class__

    def CheckQnums(self,list):
        return True
        

class Zn:
    def __init__(self,n):
        if n < 2:
            raise ValueError("Symmetry.Zn","[ERROR] discrete symmetry Zn must have n >= 2.")

        self.n = int(n)

    def CombineRule(self,A,B):
        return (A+B)%self.n

    def __repr__(self):
        return 'Z%d'%(self.n)

    def __str__(self):
        return 'Z%d'%(self.n)

    def __eq__(self,rhs):
        if (self.__class__ == rhs.__class__):
            if self.n == rhs.n:
                return True
            else:
                return False
        else:
            return False

    def CheckQnums(self,qlist):
        for q in qlist.flatten():
            if q < 0 or q >=self.n:
                return False
        return True

