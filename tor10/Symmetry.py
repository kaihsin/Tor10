import numpy as np 
import torch as tor

class U1:
    """
        U1 Symmetry class. 
        The U1 symmetry can have quantum number represent as arbitrary unsigned integer.

        Fusion rule for combine two quantum number:
            
            q1 + q2


        >>> b1 = tor10.Bond(3,tor10.BD_BRA, qnums=[[0],[0],[1 ]     ],sym_types=[tor10.Symmetry.U1()])
        >>> b2 = tor10.Bond(4,tor10.BD_BRA, qnums=[[0],[2],[-3],[-1]],sym_types=[tor10.Symmetry.U1()])
        >>> print(b1)
        Dim = 3 |
        BRA     : U1::  +1 +0 +0

        >>> print(b2)
        Dim = 4 |
        BRA     : U1::  +2 +0 -1 -3

        >>> b1.combine(b2)
        >>> print(b1)
        Dim = 12 |
        BRA     : U1::  +3 +1 +0 -2 +2 +0 -1 -3 +2 +0 -1 -3

    """
    def __init__(self):
        pass 

    def CombineRule(self,A,B):
        return A + B

    def __repr__(self):
        return 'U1'

    def __str__(self):
        return 'U1'

    def __eq__(self,rhs):
        
        return self.__class__ == rhs.__class__

    def CheckQnums(self,list):
        return True
        

class Zn:
    """
        Z(n) Symmetry class. 
        The Z(n) symmetry can have integer quantum number, with n > 1.

        Fusion rule for combine two quantum number:
            
            (q1 + q2)%n

        
        >>> b1 = tor10.Bond(3,tor10.BD_BRA, qnums=[[0],[2],[1 ]     ],sym_types=[tor10.Symmetry.Zn(4)])
        >>> b2 = tor10.Bond(4,tor10.BD_BRA, qnums=[[0],[2],[3],[1]  ],sym_types=[tor10.Symmetry.Zn(4)])
        >>> print(b1)
        Dim = 3 |
        BRA     : Z4::  +2 +1 +0

        >>> print(b2)
        Dim = 4 |
        BRA     : Z4::  +3 +2 +1 +0

        >>> b1.combine(b2)
        >>> print(b1)
        Dim = 12 |
        BRA     : Z4::  +1 +0 +3 +2 +0 +3 +2 +1 +3 +2 +1 +0


    """
    def __init__(self,n):
        if n < 2:
            raise ValueError("Symmetry.Zn","[ERROR] discrete symmetry Zn must have n >= 2.")

        self.n = int(n)

    def CombineRule(self,A,B):
        return (A+B)%self.n

    def __repr__(self):
        return 'Z%d' % self.n

    def __str__(self):
        return 'Z%d' % self.n

    def __eq__(self,rhs):
        if self.__class__ == rhs.__class__:
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

