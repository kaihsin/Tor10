from .UniTensor import *
import torch 
import numpy as np


def Hosvd(a,order,bonds_group,by_label=False,core=True):
    """
    Calculate the higher-order SVD on a UniTensor
    Args:
        a:
            UniTensor

        order:
            a python list or 1d numpy array, indicating how the bonds should be permute before hosvd

        bonds_group:
            a python list or 1d numpy array. This indicate how the bonds of the input UniTensor should be group to perform hosvd.

        by_label:
            bool, the element in argument "order" represents the index of bond or label of bond. If True, all the elements in "order" represent the labels.

        core:
            bool, if True, the coreTensor will be compute and returned.

    Return:
        if core is true, return 2d tuple, with structure (list of unitary tensors, coreTensor)

        if core is False, return a list of unitary tensors. 

    Example:
        >>> x = Tt.From_torch(tor.arange(0.1,2.5,0.1).reshape(2,3,4).to(tor.float64),labels=[6,7,8],N_inbond=1)
        >>> x.Print_diagram()
        tensor Name : 
        tensor Rank : 3
        on device   : cpu
        is_diag     : False
                ---------------     
                |             |     
            6 __| 2         3 |__ 7  
                |             |     
                |           4 |__ 8  
                |             |     
                ---------------     
        lbl:6 Dim = 2 |
        IN  :
        _
        lbl:7 Dim = 3 |
        OUT :
        _
        lbl:8 Dim = 4 |
        OUT :

        >>> print(x)
        Tensor name: 
        is_diag    : False
        tensor([[[0.1000, 0.2000, 0.3000, 0.4000],
                 [0.5000, 0.6000, 0.7000, 0.8000],
                 [0.9000, 1.0000, 1.1000, 1.2000]],
                _
                [[1.3000, 1.4000, 1.5000, 1.6000],
                 [1.7000, 1.8000, 1.9000, 2.0000],
                 [2.1000, 2.2000, 2.3000, 2.4000]]], dtype=torch.float64)

        >>> factors, core = Tt.Hosvd(x,order=[7,6,8],bonds_group=[2,1],by_label=True)
        >>> core.Print_diagram()
        tensor Name : 
        tensor Rank : 2
        on device   : cpu
        is_diag     : False
                ---------------     
                |             |     
                |           4 |__ -1 
                |             |     
                |           4 |__ -2 
                |             |     
                ---------------     
        lbl:-1 Dim = 4 |
        OUT :
        _
        lbl:-2 Dim = 4 |
        OUT :

        >>> print(len(factors))
        2

        >>> factor[0].Print_diagram()
        tensor Name : 
        tensor Rank : 3
        on device   : cpu
        is_diag     : False
                ---------------     
                |             |     
            7 __| 3         4 |__ -1 
                |             |     
            6 __| 2           |      
                |             |     
                ---------------     
        lbl:7 Dim = 3 |
        IN  :
        _
        lbl:6 Dim = 2 |
        IN  :
        _
        lbl:-1 Dim = 4 |
        OUT :
        
        >>> factor[1].Print_diagram()
        tensor Name : 
        tensor Rank : 2
        on device   : cpu
        is_diag     : False
                ---------------     
                |             |     
            8 __| 4         4 |__ -2 
                |             |     
                ---------------     
        lbl:8 Dim = 4 |
        IN  :
        _
        lbl:-2 Dim = 4 |
        OUT :


        * Checking:

        >>> rep_x = core
        >>> for f in factors:
        >>>     rep_x = Tt.Contract(rep_x,f)
        >>> rep_x.Permute([6,7,8],N_inbond=1,by_label=True)
        >>> print(rep_x - x)
        Tensor name: 
        is_diag    : False
        tensor([[[3.0531e-16, 6.1062e-16, 5.5511e-16, 4.9960e-16],
                 [6.6613e-16, 8.8818e-16, 8.8818e-16, 8.8818e-16],
                 [5.5511e-16, 4.4409e-16, 8.8818e-16, 6.6613e-16]],
                _
                [[6.6613e-16, 8.8818e-16, 1.1102e-15, 8.8818e-16],
                 [1.9984e-15, 2.2204e-15, 2.6645e-15, 1.7764e-15],
                 [1.7764e-15, 2.6645e-15, 2.6645e-15, 1.7764e-15]]],
               dtype=torch.float64)

    """
    if not isinstance(a,UniTensor):
        raise TypeError("Hosvd(UniTensor,*args)","[ERROR] the input should be a UniTensor")

    if not (isinstance(bonds_group,list) or isinstance(bonds_group,np.array)):
        raise TypeError("Hosvd(UniTensor,order,bonds_group,*args)","[ERROR] the bonds_group should be a python list or 1d numpy array")
    
    if not (isinstance(order,list) or isinstance(order,np.array)):
        raise TypeError("Hosvd(UniTensor,order,bonds_group,*args)","[ERROR] the order should be a python list or 1d numpy array")

    ## checking:
    if len(order) != len(a.labels):
        raise ValueError("Hosvd","[ERROR] the size of order should be equal to the rank of input UniTensor. size of order:%d; rank of UniTensor:%d"%(len(order),len(a.labels)))

    ## checking:
    if len(a.labels)<3:
        raise Exception("Hosvd","[ERROR], Hosvd can only perform on a UniTensor with rank > 2. For a rank-2 tensor, using Svd() instead.")

    ## checking:
    if all( x<=0 for x in bonds_group):
        
        raise ValueError("Hosvd","[ERROR] bonds_group cannot have elements <=0")

    old_labels = copy.copy(a.labels)
    old_bonds  = copy.deepcopy(a.bonds)

    if by_label:
        maper = copy.copy(order)
    else:
        if not all(id<len(a.labels) for id in order):
            raise ValueError("Hosvd","[ERROR] by_label=False but the input 'order' exceed the rank of UniTensor")
        maper = a.labels[order]
    
    

    factors = []
    start_label = np.min(a.labels)
    start_label = start_label-1 if start_label<=0 else -1
    for bg in bonds_group:
        a.Permute(maper,N_inbond=bg,by_label=True)

        ## manipulate only the Storage, keep the shell of UniTensor unchange.
        old_shape = a.Storage.shape
        a.Contiguous()
        a.Storage = a.Storage.view(np.prod(a.Storage.shape[:bg]),-1)
        u,_,_ = torch.svd(a.Storage)
        
        new_bonds = np.append(copy.deepcopy(a.bonds[:bg]),Bond(BD_OUT,u.shape[-1]))
        new_labels= np.append(copy.copy(a.labels[:bg]),start_label)

        factors.append( UniTensor(bonds=new_bonds,labels=new_labels,torch_tensor=u.view(*list(old_shape[:bg]),-1),check=False) )

        a.Storage = a.Storage.view(old_shape)
        start_label -= 1 
        maper = np.roll(maper,-bg)
 
    a.Permute(old_labels,N_inbond=1,by_label=True)
    a.bonds = old_bonds
    
    ## if compute core?

    if not core:
        return factors
    else:
        out = a    
        for n in factors:
            out = Contract(out,n)
        return factors,out
    


def Abs(a):
    """
    Take the absolute value for all the elements in the UniTensor
    Args:
        a:
            UniTensor

    Return:
        UniTensor, same shape as the input.
    """
    if not isinstance(a,UniTensor):
        raise TypeError("Abs(UniTensor)","[ERROR] the input should be a UniTensor")

    return UniTensor(bonds=copy.deepcopy(a.bonds),labels=copy.copy(a.labels),is_diag=a.is_diag,torch_tensor= torch.abs(a.Storage),check=False)

def Mean(a):
    """
    Calculate the mean of all elements in the input UniTensor

    Args:
        a: 
            UniTensor

    Return:
        UniTensor, 0-rank (constant)

    """
    if not isinstance(a,UniTensor):
        raise TypeError("Mean(UniTensor)","[ERROR] the input should be a UniTensor")

    return UniTensor(bonds=[],labels=[],torch_tensor=torch.mean(a.Storage),check=False)

def Otimes(a,b):
    """
    Perform matrix product for two rank-2 tensors.

        :math:`a \otimes b`    

    Args:
        a:  
            UniTensor, must be rank-2
        
        b:
            UUniTensor, must be rank-2

    Return:
        UniTensor, rank-2, one in-bond one out-bond. 
        If both a and b are diagonal matrix (is_diag=True), the return UniTensor will be a diagonal tensor.

        If one of the input tensor is diagonal matrix and the other is not, the return UniTensor will be densed.


    """

    if isinstance(a,UniTensor) and isinstance(b,UniTensor):
        if len(a.labels)==2 and len(b.labels)==2:
            if a.is_diag and b.is_diag:

                return UniTensor(bonds=[Bond(BD_IN,out.shape[0]),Bond(BD_OUT,out.shape[0])],\
                                 torch_tensor=torch.ger(a.Storage,b.Storage),\
                                 is_diag=True,check=False)


            if a.is_diag:
                tmpa = torch.diag(a.Storage)
            else:
                tmpa = a.Storage

            if b.is_diag:
                tmpb = torch.diag(b.Storage)
            else:
                tmpb = b.Storage            

            out = torch.tensordot(a.Storage,b.Storage,dims=0).permute(0,2,1,3).reshape(a.Storage.shape[0]*b.Storage.shape[0],-1)
            return UniTensor(bonds=[Bond(BD_IN,out.shape[0]),Bond(BD_OUT,out.shape[1])],\
                             torch_tensor=out,\
                             check=False)

        else:
            raise TypeError("Otimes","[ERROR], Otimes only accept rank-2 UniTensors as arguments.")
        
    else:
        raise TypeError("Otimes","[ERROR], Otimes only accept UniTensor as arguments.")



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

        :math:`a = u \cdot s \cdot vt`

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


    Example:
    ::
        y = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,3),Tt.Bond(Tt.BD_OUT,4)])
        y.SetElem([1,1,0,1,
                   0,0,0,1,
                   1,1,0,0]


    >>> print(y)
    Tensor name: 
    is_diag    : False
    tensor([[1., 1., 0., 1.],
            [0., 0., 0., 1.],
            [1., 1., 0., 0.]], dtype=torch.float64)

    >>> u,s,v = Tt.linalg.Svd(y)
    >>> print(u)
    Tensor name: 
    is_diag    : False
    tensor([[-0.7887, -0.2113, -0.5774],
            [-0.2113, -0.7887,  0.5774],
            [-0.5774,  0.5774,  0.5774]], dtype=torch.float64)

    >>> print(s)
    Tensor name: 
    is_diag    : True
    tensor([2.1753e+00, 1.1260e+00, 1.0164e-16], dtype=torch.float64)

    >>> print(v)
    Tensor name: 
    is_diag    : False
    tensor([[-6.2796e-01, -6.2796e-01,  0.0000e+00, -4.5970e-01],
            [ 3.2506e-01,  3.2506e-01,  0.0000e+00, -8.8807e-01],
            [-7.0711e-01,  7.0711e-01,  0.0000e+00,  1.1309e-16]],
            dtype=torch.float64)

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



def Svd_truncate(a, keepdim=None):
    """
    The function performs the svd to input UniTensor, and truncate [truncate] dim from the smallest singular value to the tensor. The UniTensor should be rank-2. each bond's dim should be >=2. 


    Args:
        a : 
            UniTensor, rank-2, 1 inbond 1 outbond.
    
        keepdim:
            integer, the keeping dimension. When set, it will keep only the largest "keepdim" singular values and their corresponding eigenvectors.


    Return: u , s , vt 
        u : 
            UniTensor, 2-rank, 1 inbond 1 outbond, the truncated unitary matrix with shape (a.shape()[0], truncate)
        
        s : 
            UniTensor, 2-rank, 1 inbond 1 outbond, the diagonal, truncated singular matrix with shape (truncate,truncate)
                        
        vt: 
            UniTensor, 2-rank, 1 inbond 1 outbond, the transposed right unitary matrix with shape (truncate,a.shape()[1])


    Example:
    ::
        y = Tt.UniTensor(bonds=[Tt.Bond(Tt.BD_IN,3),Tt.Bond(Tt.BD_OUT,4)])
        y.SetElem([1,1,0,1,
                   0,0,0,1,
                   1,1,0,0])

    >>> print(y)
    Tensor name: 
    is_diag    : False
    tensor([[1., 1., 0., 1.],
            [0., 0., 0., 1.],
            [1., 1., 0., 0.]], dtype=torch.float64)

    >>> u,s,v = Tt.linalg.Svd_truncate(y,keepdim=2)
    >>> print(u)
    Tensor name: 
    is_diag    : False
    tensor([[-0.7887, -0.2113],
            [-0.2113, -0.7887],
            [-0.5774,  0.5774]], dtype=torch.float64)
 
    >>> print(s)
    Tensor name: 
    is_diag    : True
    tensor([2.1753, 1.1260], dtype=torch.float64)

    >>> print(v)
    Tensor name: 
    is_diag    : False
    tensor([[-0.6280, -0.6280,  0.0000, -0.4597],
            [ 0.3251,  0.3251,  0.0000, -0.8881]], dtype=torch.float64)

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

        if keepdim is not None:
            if keepdim < 0 or keepdim > len(s):
                raise ValueError("Svd_truncate", "[ERROR] the keepdim=%d is invalid, must larger than 0 and smaller than the total number of eigenvalues."%(keepdim))
            u = u[:, :keepdim]
            s = s[:keepdim]
            v = v[:, :keepdim]

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
        raise Exception("Svd_truncate(UniTensor,int)","[ERROR] Svd_truncate can only accept UniTensor")

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
        UniTensor, 0-rank (constant)

    Example:
    ::
        a = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,3)])
        a.SetElem([4,-3,0,
                   2,-1,2,
                   1, 5,7])
        b = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,3),Tor10.Bond(Tor10.BD_OUT,3)],is_diag=True)
        b.SetElem([1,2,3])

    >>> print(a)
    Tensor name: 
    is_diag    : False
    tensor([[ 4., -3.,  0.],
            [ 2., -1.,  2.],
            [ 1.,  5.,  7.]], dtype=torch.float64)

    >>> out = Tt.Det(a)
    >>> print(out)
    Tensor name: 
    is_diag    : False
    tensor(-32., dtype=torch.float64)

    >>> print(b)
    Tensor name: 
    is_diag    : True
    tensor([1., 2., 3.], dtype=torch.float64)

    >>> out = Tor10.Det(b)
    >>> print(out)
    Tensor name: 
    is_diag    : False
    tensor(6., dtype=torch.float64)
                
    """
    if isinstance(a,UniTensor):

        if a.is_diag:
            tmp = torch.prod(a.Storage)
        else:
            tmp = torch.det(a.Storage)
    
        return UniTensor(bonds=[],labels=[],torch_tensor=tmp,check=False)

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
        UniTensor, 0-rank (constant)
                    
    """

    if isinstance(a,UniTensor):
        return UniTensor(bonds=[],labels=[],torch_tensor=torch.norm(a.Storage),check=False)
    else:
        raise Exception("Norm(UniTensor)","[ERROR] Norm can only accept UniTensor")
