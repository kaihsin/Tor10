from .UniTensor import *
import torch 
import numpy as np

def ExpH(a):
    """
    This function performs 

            :math:`e^{H}`

    where H is the hermitian matrix. 
    The Intricate computation follows procedure: symeig() -> exp() the singular matrix.

    Args:
        
        a : 
            UniTensor, Must be a rank-2. If pass a non-rank2 tensor or pass a non-hermitian rank2 tensor, it will raise Error.  

    Return:

        UniTensor, 2-rank, same bonds and labels as the original H
    """

    if isinstance(a,UniTensor):
        ## Qnum_ipoint
        if a.bonds[0].qnums is not None:
            raise Exception("ExpH(a)","[Abort] curretly don't support symm tensor.")

        if a.is_diag:
            u = torch.exp(a.Storage)
            return UniTensor(bonds=a.bonds,\
                             labels=a.labels,\
                             torch_tensor=u,\
                             is_diag=True,\
                             check=False)
        else:
            ## version-1, only real, not sure if it can extend to complex
            s , u = torch.symeig(a.Storage,eigenvectors=True)
            s     = torch.exp(s)

            # torch.matmul(u*s,u.transpose(0,1),out=u)
            u = torch.matmul(u*s,u.transpose(0,1))
            del s

            return UniTensor(bonds=a.bonds,\
                            labels=a.labels,\
                            torch_tensor=u,\
                            check=False)
                
    else:
        raise Exception("ExpH(UniTensor)","[ERROR] ExpH can only accept UniTensor")



def Qr(a):
    """
    The function performs the qr decomposition 

        :math:`a = q \cdot r`

    to the input UniTensor. The UniTensor should be rank-2. each bond's dim should be >=2. 

    
    Args:

        a : UniTensor, it is required to be a non-diagonal rank-2 tensor. If pass a non rank-2 tensor or diagonal matrix, it will throw Exception.

    Return:
        
        q , r  
        
        q : UniTensor, 2-rank, 1 inbond 1 outbond, the unitary matrix
        
        r : UniTensor, 2-rank, 1 inbond 1 outbond, the upper triangular matrix 

    """
    if isinstance(a,UniTensor):

        ## Qnum_ipoint
        if a.bonds[0].qnums is not None:
            raise Exception("Qr(a)","[Abort] curretly don't support symm tensor.")

        if a.is_diag:
            raise Exception("Qr(UniTensor)","[Aboart] Currently not support diagonal tensors.")
        
        q, r = torch.qr(a.Storage)

        tmp = np.argwhere(a.labels<0)
        if len(tmp) == 0:
            tmp = 0
        else:
            tmp = np.min(tmp)

        q = UniTensor(bonds =[Bond(BD_IN,q.shape[0]),Bond(BD_OUT,q.shape[1])],\
                      labels=[a.labels[0],tmp-1],\
                      torch_tensor=q,\
                      check=False)
        r = UniTensor(bonds =[Bond(BD_IN,r.shape[0]),Bond(BD_OUT,r.shape[1])],\
                      labels=[q.labels[1],a.labels[1]],\
                      torch_tensor=r,\
                      check=False)
        return q,r
    else:
        raise Exception("Qr(UniTensor)","[ERROR] Qr can only accept UniTensor")


def Qdr(a):
    """
    The function performs the qdr decomposition 

        :math:`a = q \cdot d \cdot r`

    to input UniTensor. The UniTensor should be rank-2 with eachbond's dim should be >=2. 
    
    Args:
        a : 
            UniTensor , rank-2, 1 inbond 1 outbond.

    Return: q , r  
        q : 
            UniTensor, 2-rank, 1 inbond 1 outbond, the unitary matrix

        d :
            The diagonal matrix. It is a diagonal 2-rank UniTensor with 1 inbond 1 outbond and is_diag=True.
        r : 
            UniTensor, 2-rank, 1 inbond 1 outbond, the upper triangular matrix 
    """
    if isinstance(a,UniTensor):

        ## Qnum_ipoint
        if a.bonds[0].qnums is not None:
            raise Exception("Qdr(a)","[Abort] curretly don't support symm tensor.")

        if a.is_diag:
            raise Exception("Qr(UniTensor)","[Aboart] Currently not support diagonal tensors.")

        q, r = torch.qr(a.Storage)
        d = r.diag()
        r = (r.t()/d).t()

        tmp = np.argwhere(a.labels<0)
        if len(tmp) == 0:
            tmp = 0
        else:
            tmp = np.min(tmp)

        q = UniTensor(bonds =[Bond(BD_IN,q.shape[0]),Bond(BD_OUT,q.shape[1])],\
                      labels=[a.labels[0],tmp-1],\
                      torch_tensor=q,\
                      check=False)
        d = UniTensor(bonds =[Bond(BD_IN,d.shape[0]),Bond(BD_OUT,d.shape[0])],\
                      labels=[tmp-1,tmp-2],\
                      torch_tensor=d,\
                      is_diag=True,
                      check=False)
        r = UniTensor(bonds =[Bond(BD_IN,r.shape[0]),Bond(BD_OUT,r.shape[1])],\
                      labels=[d.labels[1],a.labels[1]],\
                      torch_tensor=r,\
                      check=False)
        return q,d,r
    else:
        raise Exception("Qdr(UniTensor)","[ERROR] Qdr can only accept UniTensor")

def Svd(a):
    """
    The function performs the svd 

        :math:`a = u \cdot s \ cdot vt`

    to input UniTensor. The UniTensor should be rank-2. each bond's dim should be >=2. 

    Args:
        a : 
            UniTensor, rank-2.

    Return: u , s , vt 
        u : 
            UniTensor, 2-rank, 1 inbond 1 outbond, the unitary matrix
                        
        s : 
            UniTensor, 2-rank, 1 inbond 1 outbond, the diagonal, singular matrix, with is_diag=True
 
        vt: 
            UniTensor, 2-rank, 1 inbond 1 outbond, the transposed right unitary matrix

    """
    if isinstance(a,UniTensor):

        ## Qnum_ipoint
        if a.bonds[0].qnums is not None:
            raise Exception("svd(a)","[Abort] svd curretly don't support symm tensor.")


        if a.is_diag:
            raise Exception("svd(a)","[Abort] svd currently don't support diagonal tensor.")

        u, s, v = torch.svd(a.Storage,some=True)

        tmp = np.argwhere(a.labels<0)
        if len(tmp) == 0:
            tmp = 0
        else:
            tmp = np.min(tmp)

        u = UniTensor(bonds =[Bond(BD_IN,u.shape[0]),Bond(BD_OUT,u.shape[1])],\
                      labels=[a.labels[0],tmp-1],\
                      torch_tensor=u,\
                      check=False)
        v = UniTensor(bonds =[Bond(BD_IN,v.shape[1]),Bond(BD_OUT,v.shape[0])],\
                      labels=[tmp-2,a.labels[1]],\
                      torch_tensor=v.transpose(0,1),\
                      check=False)
        s = UniTensor(bonds  =[u.bonds[1],v.bonds[0]],\
                      labels =[u.labels[1],v.labels[0]],\
                      torch_tensor=s,\
                      check=False,\
                      is_diag=True)   
        return u,s,v
    else:
        raise Exception("Svd(UniTensor)","[ERROR] Svd can only accept UniTensor")



def Svd_truncate(a, truncate=None):
    """
    The function performs the svd to input UniTensor, and truncate [truncate] dim from the smallest singular value to the tensor. The UniTensor should be rank-2. each bond's dim should be >=2. 

    Args:
        a : 
            UniTensor, rank-2, 1 inbond 1 outbond.
    
    Return: u , s , vt 
        u : 
            UniTensor, 2-rank, 1 inbond 1 outbond, the truncated unitary matrix with shape (a.shape()[0], truncate)
        
        s : 
            UniTensor, 2-rank, 1 inbond 1 outbond, the diagonal, truncated singular matrix with shape (truncate,truncate)
                        
        vt: 
            UniTensor, 2-rank, 1 inbond 1 outbond, the transposed right unitary matrix with shape (truncate,a.shape()[1])

    """
    if isinstance(a,UniTensor):

        ## Qnum_ipoint
        if a.bonds[0].qnums is not None:
            raise Exception("Qdr(a)","[Abort] curretly don't support symm tensor.")

        if a.is_diag:
            raise Exception("svd(a)","[Abort] svd currently don't support diagonal tensor.")

        u, s, v = torch.svd(a.Storage,some=True)

        tmp = np.argwhere(a.labels<0)
        if len(tmp) == 0:
            tmp = 0
        else:
            tmp = np.min(tmp)

        if truncate is not None:
            if truncate < 0 or truncate > len(s):
                raise ValueError("Svd_truncate", "[ERROR] the truncate dimension is invalid")
            u = u[:, :truncate]
            s = s[:truncate]
            v = v[:, :truncate]

        u = UniTensor(bonds =[Bond(BD_IN,u.shape[0]),Bond(BD_OUT,u.shape[1])],\
                      labels=[a.labels[0],tmp-1],\
                      torch_tensor=u,\
                      check=False)
        v = UniTensor(bonds =[Bond(BD_IN,v.shape[1]),Bond(BD_OUT,v.shape[0])],\
                      labels=[tmp-2,a.labels[1]],\
                      torch_tensor=v.transpose(0,1),\
                      check=False)
        s = UniTensor(bonds  =[u.bonds[1],v.bonds[0]],\
                      labels =[u.labels[1],v.labels[0]],\
                      torch_tensor=s,\
                      check=False,\
                      is_diag=True)   
        return u,s,v
    else:
        raise Exception("Svd(UniTensor)","[ERROR] Svd can only accept UniTensor")

def Matmul(a,b):
    """
    Performs matrix multiplication on the rank-2 UniTensors. 

        :math:`A \cdot B`

    Note that both the UniTensors should be rank-2, and dimension should be matched. 

    If a and b are both diagonal matrix, the return will be a diagonal matrix. If one (or both) of them are non-diagonal matrix and the other is diagonal matrix, the return will be a dense matrix.

    Args:
        a: 
            The UniTensors that will be matrix-multiply

        b: 
            The UniTensors that will be matrix-multiply

    Return:
        UniTensor,rank-2 tensor with 1 inbond 1 outbond. 

    """
    if isinstance(a,UniTensor) and isinstance(b,UniTensor):

        ## [Note] no need to check if a,b are both rank 2. Rely on torch to do error handling! 

        ## Qnum_ipoint
        if a.bonds[0].qnums is not None or b.bonds[0].qnums is not None:
            raise Exception("Matmul(a,b)","[Abort] Matmul for sym TN is under developing.")

        if a.is_diag == b.is_diag:
            tmp = UniTensor(bonds =[a.bonds[0],b.bonds[1]],\
                            torch_tensor=torch.matmul(a.Storage,b.Storage),\
                            check=False,\
                            is_diag=a.is_diag)
        else:
            if a.is_diag:
                tmp = UniTensor(bonds =[a.bonds[0],b.bonds[1]],\
                                torch_tensor=torch.matmul(torch.diag(a.Storage),b.Storage),\
                                check=False)
            if b.is_diag:
                tmp = UniTensor(bonds =[a.bonds[0],b.bonds[1]],\
                                torch_tensor=torch.matmul(a.Storage,torch.diag(b.Storage)),\
                                check=False)

        return tmp

    else:
        raise TypeError("_Matmul(a,b)", "[ERROR] _Matmul can only accept UniTensors for both a & b")


def Chain_matmul(*args):
    """
    Performs matrix multiplication on all the UniTensors. 

        :math:`A \cdot B \cdot C \cdot D \cdots`

    Note that 
    
    1. all the UniTensors should be rank-2, and dimension should be matched. 
    
    2. The input UniTensors can have some of them are diagonal matrix (is_diag=True). The return will always be a rank-2 UniTensor with is_diag=False 

    Args:
        *args: 
            The UniTensors that will be matrix-multiply

    Return:
        UniTensor,rank-2 tensor with 1 inbond 1 outbond. 

    Example:
    ::
        a = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,4)])
        b = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,4),Tor10.Bond(Tor10.BD_OUT,5)])
        c = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,5),Tor10.Bond(Tor10.BD_OUT,6)])   
        d = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,6),Tor10.Bond(Tor10.BD_OUT,2)])

    >>> f = Tor10.Chain_matmul(a,b,c,d)
    >>> f.Print_diagram()
    tensor Name : 
    tensor Rank : 2
    on device   : cpu
    is_diag     : False
            ---------------     
            |             |     
        0 __| 3         2 |__ 1  
            |             |     
            ---------------     
    lbl:0 Dim = 3 |
    IN :
    _
    lbl:1 Dim = 2 |
    OUT :

    """
    f = lambda x,idiag: torch.diag(x) if idiag else x 
    isUT = all( isinstance(UT,UniTensor) for UT in args)    
    
    
    tmp_args = [f(args[i].Storage,args[i].is_diag) for i in range(len(args))] 

    ## Checking performance:
    #"""  
    #for i in range(len(tmp_args)):
    #    if not tmp_args[i] is args[i].Storage:
    #       print("Fatal performance")
    #       exit(1) 
    #"""

    if isUT:
        ## Qnum_ipoint
        if not all( (UT.bonds[0].qnums is None) for UT in args):
            raise Exception("Chain_matmul(*args)","[Abort] Chain multiplication for symm tensor(s) are under developing.")


        return UniTensor(bonds =[args[0].bonds[0],args[-1].bonds[1]],\
                         torch_tensor=torch.chain_matmul(*tmp_args),\
                         check=False)

    else:
        raise TypeError("_Chain_matmul(*args)", "[ERROR] _Chain_matmul can only accept UniTensors for all elements in args")






def Inverse(a):
    """
    This function returns the inverse of a rank-2 tensor (matrix).
    
        :math:`a^{-1}`

    If the input UniTensor is diagonal, the return will also be a diagonal matrix.

    Args:
        a : 
            A rank-2 UniTensor (matrix). Note that if the matrix is not inversable, error will be issued. passing a non-rank2 UniTensor, error will be issued. 

    Return:
        UniTensor
                    
    """
    if isinstance(a,UniTensor):
        
        if a.is_diag:
            a_inv = UniTensor(bonds = a.bonds,
                          labels=a.labels,
                          torch_tensor=a.Storage**-1,
                          is_diag=True,
                          check=False)
        else:
            a_inv = UniTensor(bonds = a.bonds,
                              labels=a.labels,
                              torch_tensor=torch.inverse(a.Storage),
                              check=False)
        return a_inv
    else:
        raise Exception("Inverse(UniTensor)","[ERROR] Inverse can only accept UniTensor")


def Det(a):
    """
    This function returns the determinant a rank-2 tensor.
    
    :math:`\det(a)`

    Args:
        a : 
            a rank-2 UniTensor (matrix). 
    Return:
        constant

    Example:
    ::
        a = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,3),Tt.Bond(Tt.BD_OUT,3)])
        a.SetElem([4,-3,0,
                   2,-1,2,
                   1, 5,7])
        b = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,3),Tt.Bond(Tt.BD_OUT,3)],is_diag=True)
        b.SetElem([1,2,3])

    >>> print(a)
    Tensor name: 
    is_diag    : False
    tensor([[ 4., -3.,  0.],
            [ 2., -1.,  2.],
            [ 1.,  5.,  7.]], dtype=torch.float64)

    >>> out = Tt.Det(a)
    >>> print(out)
    -32.0

    >>> print(b)
    Tensor name: 
    is_diag    : True
    tensor([1., 2., 3.], dtype=torch.float64)

    >>> out = Tor10.Det(b)
    >>> print(out)
    6.0

                    
    """
    if isinstance(a,UniTensor):

        if a.is_diag:
            return torch.prod(a.Storage).item()
        else:
            return torch.det(a.Storage).item()
    else:
        raise Exception("Det(UniTensor)","[ERROR] Det can only accept UniTensor")

def Norm(a):
    """
    Returns the matrix norm of the UniTensor. 

    If the given UniTensor is a matrix (rank-2), matrix norm will be calculated. If the given UniTensor is a vector (rank-1), vector norm will be calculated. If the given UniTensor has more than 2 ranks, the vector norm will be appllied to last dimension. 

    Args:
        a : 
            a UniTensor.

    Return:
        constant
                    
    """

    if isinstance(a,UniTensor):
        return torch.norm(a.Storage).item()
    else:
        raise Exception("Norm(UniTensor)","[ERROR] Norm can only accept UniTensor")
