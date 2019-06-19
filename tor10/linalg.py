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
        >>> x = tor10.From_torch(torch.arange(0.1,2.5,0.1).reshape(2,3,4).to(torch.float64),labels=[6,7,8],rowrank=1)
        >>> x.Print_diagram()

        >>> print(x)

        >>> factors, core = tor10.Hosvd(x,order=[7,6,8],bonds_group=[2,1],by_label=True)
        >>> core.Print_diagram()

        >>> print(len(factors))

        >>> factor[0].Print_diagram()


        >>> factor[1].Print_diagram()

        * Checking:

        >>> rep_x = core
        >>> for f in factors:
        >>>     rep_x = tor10.Contract(rep_x,f)
        >>> rep_x.Permute([6,7,8],rowrank=1,by_label=True)
        >>> print(rep_x - x)

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



    ## master switch
    if a.is_symm :
        raise Exception("Hosvd can only operate on non-symmetry tensor")

    else:
    
        old_labels = copy.copy(a.labels)
        old_bonds  = copy.deepcopy(a.bonds)

        if by_label:
            mapper = copy.copy(order)
        else:
            if not all(id<len(a.labels) for id in order):
                raise ValueError("Hosvd","[ERROR] by_label=False but the input 'order' exceed the rank of UniTensor")
            mapper = a.labels[order]

        iod = [np.argwhere(a.labels==mapper[x])[0]>=a.rowrank for x in range(len(mapper))]
        old_Nin = a.rowrank
        factors = []
        start_label = np.min(a.labels)
        start_label = start_label-1 if start_label<=0 else -1
        for bg in bonds_group:
            a._Permute(mapper,rowrank=bg,by_label=True)

            ## manipulate only the Storage, keep the shell of UniTensor unchange.
            old_shape = a.Storage.shape
            a.Contiguous()
            a.Storage = a.Storage.view(np.prod(a.Storage.shape[:bg]),-1)
            u,_,_ = torch.svd(a.Storage)

            new_bonds = np.append(copy.deepcopy(a.bonds[:bg]),Bond(u.shape[-1]))
            new_labels= np.append(copy.copy(a.labels[:bg]),start_label)
            iiod = np.append(iod[:bg],1)
            tmpt = UniTensor(bonds=new_bonds,labels=new_labels,rowrank=1,check=False)
            tmpt._mac(torch_tensor = u.view(*list(old_shape[:bg]),-1))
            factors.append( tmpt )
            x = np.argsort(iiod)       
            factors[-1]._Permute(x,rowrank=len(np.where(iiod==0)[0]))

            a.Storage = a.Storage.view(old_shape)
            start_label -= 1
            mapper = np.roll(mapper,-bg)
            iod   = np.roll(iod,-bg)

        a._Permute(old_labels,rowrank=old_Nin,by_label=True)
        a.bonds = old_bonds

        ## if compute core?

        if not core:
            return factors
        else:
            out = a
            for n in factors:
                n.Whole_transpose()
                out = Contract(out,n)
                n.Whole_transpose()
            
            return factors,out



def Abs(a):
    ## v0.3 OK
    """
    Take the absolute value for all the elements in the UniTensor
    Args:
        a:
            UniTensor, can be [untagged][tagged][symm]

    Return:
        UniTensor, same shape and type as the input.
    """
    if not isinstance(a,UniTensor):
        raise TypeError("Abs(UniTensor)","[ERROR] the input should be a UniTensor")

    
    if a.is_symm:
        tmp =  UniTensor(bonds=a.bonds,\
                         rowrank=a.rowrank,\
                         labels = a.labels,\
                         check=False)
        tmp._mac(torch_tensor = [torch.abs(self.Storage[b]) for b in range(len(self.Storage))],\
                           braket = a.braket,\
                           sym_mappers=(self._mapper,self._inv_mapper,\
                                      self._bra_mapper_blks,self._bra_invmapper_blks,\
                                      self._ket_mapper_blks,self._ket_invmapper_blks,\
                                      self._contiguous,self._accu_off_in,self._accu_off_out))
                         
    else:
        tmp =  UniTensor(bonds=a.bonds,\
                         rowrank=a.rowrank,\
                         labels=a.labels,\
                         is_diag=a.is_diag,\
                         check=False)
        tmp._mac(braket = a.braket,\
                            torch_tensor = torch.abs(a.Storage))
    return tmp
def Mean(a):
    ## v0.3 OK
    """
    Calculate the mean of all elements in the input non-symmetry UniTensor

    Args:
        a:
            UniTensor, can be [untagged][tagged]

    Return:
        UniTensor, 0-rank (constant)

    """
    if not isinstance(a,UniTensor):
        raise TypeError("Mean(UniTensor)","[ERROR] the input should be a UniTensor")
    if a.is_symm:
        raise Exception("Mean(UniTensor)","[ERROR] cannot get mean for a symmetry tensor. GetBlock first")
    
    tmp = UniTensor(bonds=[],labels=[],rowrank=0,check=False)
    tmp._mac(torch_tensor = torch.mean(a.Storage))
    return tmp
def Otimes(a,b):
    """
    Perform matrix product for two rank-2 tensors.

        :math:`a \otimes b`

    Args:
        a:
            UniTensor, must be rank-2, [untagged], with 1-inbond and 1-outbond

        b:
            UniTensor, must be rank-2, [untagged], with 1-inbond and 1-outbond

    Return:
        UniTensor, rank-2, one in-bond one out-bond. [untagged]
        If both a and b are diagonal matrix (is_diag=True), the return UniTensor will be a diagonal tensor.

        If one of the input tensor is diagonal matrix and the other is not, the return UniTensor will be densed.


    """

    if isinstance(a,UniTensor) and isinstance(b,UniTensor):

        if a.braket is not None or b.braket is not None:
            raise TypeError("linalg.Otimes","Otimes can only accept untagged tensor.") 

        if len(a.labels)==2 and len(b.labels)==2 and a.rowrank==1 and b.rowrank==1:
            if a.is_diag and b.is_diag:

                tmp =  UniTensor(bonds=[Bond(out.shape[0]),Bond(out.shape[0])],\
                                 rowrank=1,\
                                 is_diag=True,check=False)
                tmp._mac(torch_tensor = torch.ger(a.Storage,b.Storage))
                return tmp

            if a.is_diag:
                tmpa = torch.diag(a.Storage)
            else:
                tmpa = a.Storage

            if b.is_diag:
                tmpb = torch.diag(b.Storage)
            else:
                tmpb = b.Storage

            out = torch.tensordot(a.Storage,b.Storage,dims=0).permute(0,2,1,3).reshape(a.Storage.shape[0]*b.Storage.shape[0],-1)
            tmp =  UniTensor(bonds=[Bond(out.shape[0]),Bond(out.shape[1])],\
                             rowrank=1,\
                             check=False)
            tmp._mac(torch_tensor=out)
            return tmp

        else:
            raise TypeError("Otimes","[ERROR], Otimes only accept rank-2 UniTensors as arguments.")

    else:
        raise TypeError("Otimes","[ERROR], Otimes only accept UniTensor as arguments.")



def ExpH(a):
    # v0.3 OK
    """
    This function performs

            :math:`e^{H}`

    where H is the hermitian matrix.
    The Intricate computation follows procedure: symeig() -> exp() the singular matrix.

    Args:

        a :
            UniTensor, Must be a rank-2 [untagged], with one inbond, one outbond. If pass a non-rank2 tensor, tagged tensor, or pass a non-hermitian rank2 tensor; it will raise Error.

    Return:

        UniTensor, rank-2 [unregular], same bonds and labels and braket form as the original H
    """

    if isinstance(a,UniTensor):
        
        if a.is_symm:
           raise Exception("ExpH(a)","don't support symm tensor. GetBlock first.")

        else:
            if a.braket is not None:
                raise Exception("ExpH(a)","can only accept [untagged] type UniTensor")

            if a.is_diag:   
                u = torch.exp(a.Storage)
                
                tmp =  UniTensor(bonds=a.bonds,\
                                 labels=a.labels,\
                                 rowrank=a.rowrank,\
                                 is_diag=True,\
                                 check=False)
                tmp._mac(torch_tensor = u)
                return tmp
            else:
                if a.rowrank != 1:
                    raise Exception("ExpH(a)","a should be rank-2 tensor with 1 inbond 1 outbond")

                ## version-1, only real, not sure if it can extend to complex
                s , u = torch.symeig(a.Storage,eigenvectors=True)
                s     = torch.exp(s)

                # torch.matmul(u*s,u.transpose(0,1),out=u)
                u = torch.matmul(u*s,u.transpose(0,1))
                del s

                tmp =  UniTensor(bonds=a.bonds,\
                                labels=a.labels,\
                                rowrank=a.rowrank,\
                                check=False)
                tmp._mac(torch_tensor = u)
                return tmp

    else:
        raise Exception("ExpH(UniTensor)","[ERROR] ExpH can only accept UniTensor")



def Qr(a):
    # v0.3 OK
    """
    The function performs the qr decomposition

        :math:`a = q \cdot r`

    to the input UniTensor. The UniTensor should be rank-2, untagged. each bond's dim should be >=2.


    Args:

        a : UniTensor[untagged], it is required to be a non-diagonal rank-2 tensor. If pass a non rank-2 tensor or diagonal matrix, it will throw Exception.

    Return:

        q , r

        q : UniTensor[regular], rank-2, 1 inbond 1 outbond, the unitary matrix

        r : UniTensor[regular], rank-2, 1 inbond 1 outbond, the upper triangular matrix

    """
    if isinstance(a,UniTensor):

        ## Qnum_ipoint
        if a.is_symm:
            raise Exception("Qr(a)","don't support symm tensor. GetBlock() first")

        if a.is_diag:
            raise Exception("Qr(UniTensor)","[Aboart] Currently not support diagonal tensors.")

        if a.braket is not None:
            raise Exception("Qr(UniTensor)","Can only accept [untagged] tensor")

        if a.rowrank != 1:
            raise Exception("Qr(UniTensor)","Should have 1 in-bond, 1 out-bond")

        q, r = torch.qr(a.Storage)

        tmp = np.argwhere(a.labels<0)
        if len(tmp) == 0:
            tmp = 0
        else:
            tmp = np.min(tmp)

        tq = UniTensor(bonds =[Bond(q.shape[0]),Bond(q.shape[1])],\
                      rowrank=1,\
                      labels=[a.labels[0],tmp-1],\
                      check=False)
        tq._mac(torch_tensor=q)

        tr = UniTensor(bonds =[Bond(r.shape[0]),Bond(r.shape[1])],\
                      rowrank=1,\
                      labels=[tq.labels[1],a.labels[1]],\
                      check=False)
        tr._mac(torch_tensor = r)

        return tq,tr
    else:
        raise Exception("Qr(UniTensor)","[ERROR] Qr can only accept UniTensor")


def Qdr(a):
    # v0.3 OK
    """
    The function performs the qdr decomposition

        :math:`a = q \cdot d \cdot r`

    to input UniTensor. The UniTensor should be rank-2, untagged. with eachbond's dim should be >=2.

    Args:
        a :
            UniTensor [untagged], rank-2, 1 inbond 1 outbond.

    Return: q , r
        q :
            UniTensor [untagged], rank-2, 1 inbond 1 outbond, the unitary matrix

        d :
            The diagonal matrix [untagged]. It is a diagonal rank-2 UniTensor with 1 inbond 1 outbond and is_diag=True.
        r :
            UniTensor [untagged], rank-2, 1 inbond 1 outbond, the upper triangular matrix
    """
    if isinstance(a,UniTensor):

        ## Qnum_ipoint
        if a.is_symm:
            raise Exception("Qdr(a)","[Abort] curretly don't support symm tensor.")

        if a.is_diag:
            raise Exception("Qr(UniTensor)","[Aboart] Currently not support diagonal tensors.")

        if a.braket is not None:
            raise Exception("Qr(UniTensor)","can only operate on [regular](untagged) tensor")
            
        if a.rowrank != 1:
            raise Exception("Qdr(a)","Should have 1 inbond 1 outbond")

        
        q, r = torch.qr(a.Storage)
        d = r.diag()
        r = (r.t()/d).t()

        tmp = np.argwhere(a.labels<0)
        if len(tmp) == 0:
            tmp = 0
        else:
            tmp = np.min(tmp)

        tq = UniTensor(bonds =[Bond(q.shape[0]),Bond(q.shape[1])],\
                      rowrank=1,\
                      labels=[a.labels[0],tmp-1],\
                      check=False)
        tq._mac(torch_tensor = q)

        td = UniTensor(bonds =[Bond(d.shape[0]),Bond(d.shape[0])],\
                      rowrank=1,\
                      labels=[tmp-1,tmp-2],\
                      is_diag=True,
                      check=False)
        td._mac(torch_tensor = d)

        tr = UniTensor(bonds =[Bond(r.shape[0]),Bond(r.shape[1])],\
                      rowrank=1,\
                      labels=[td.labels[1],a.labels[1]],\
                      check=False)
        tr._mac(torch_tensor = r)

        return tq,td,tr
    else:
        raise Exception("Qdr(UniTensor)","[ERROR] Qdr can only accept UniTensor")

def Svd(a):
    """
    The function performs the svd

        :math:`a = u \cdot s \cdot vt`

    to input UniTensor. The UniTensor should be rank-2,untagged. each bond's dim should be >=2.

    Args:
        a :
            UniTensor[untagged], rank-2.

    Return: u , s , vt
        u :
            UniTensor[untagged], rank-2, 1 inbond 1 outbond, the unitary matrix

        s :
            UniTensor[untagged], rank-2, 1 inbond 1 outbond, the diagonal, singular matrix, with is_diag=True

        vt:
            UniTensor[untagged], rank-2, 1 inbond 1 outbond, the transposed right unitary matrix


    Example:
    ::
        y = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(4)],rowrank=1)
        y.SetElem([1,1,0,1,
                   0,0,0,1,
                   1,1,0,0]


    >>> print(y)
    Tensor name:
    is_diag    : False
    tensor([[1., 1., 0., 1.],
            [0., 0., 0., 1.],
            [1., 1., 0., 0.]], dtype=torch.float64)

    >>> u,s,vt = tor10.linalg.Svd(y)
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

    >>> print(vt)
    Tensor name:
    is_diag    : False
    tensor([[-6.2796e-01, -6.2796e-01,  0.0000e+00, -4.5970e-01],
            [ 3.2506e-01,  3.2506e-01,  0.0000e+00, -8.8807e-01],
            [-7.0711e-01,  7.0711e-01,  0.0000e+00,  1.1309e-16]],
            dtype=torch.float64)

    """
    if isinstance(a,UniTensor):

        ## Qnum_ipoint
        if a.is_symm:
            raise Exception("svd(a)","[Abort] svd curretly don't support symm tensor.")


        if a.is_diag:
            raise Exception("svd(a)","[Abort] svd currently don't support diagonal tensor.")

        if a.braket is not None:
            raise Exception("svd(a)","can only accept UniTensor[regular] (untagged)")

        if a.rowrank != 1:
            raise Exception("svd(a)","should be a UniTensor with 1 in-bond, 1 out-bond")

        u, s, v = torch.svd(a.Storage,some=True)

        tmp = np.argwhere(a.labels<0)
        if len(tmp) == 0:
            tmp = 0
        else:
            tmp = np.min(tmp)

       

        tu = UniTensor(bonds =[Bond(u.shape[0]),Bond(u.shape[1])],\
                      rowrank=1,\
                      labels=[a.labels[0],tmp-1],\
                      check=False)
        tu._mac(torch_tensor=u)

        tv = UniTensor(bonds =[Bond(v.shape[1]),Bond(v.shape[0])],\
                      rowrank=1,\
                      labels=[tmp-2,a.labels[1]],\
                      check=False)
        tv._mac(torch_tensor=v.transpose(0,1))

        ts = UniTensor(bonds  =[tu.bonds[1],tv.bonds[0]],\
                      labels =[tu.labels[1],tv.labels[0]],\
                      rowrank=1,\
                      check=False,\
                      is_diag=True)
        ts._mac(torch_tensor=s)

        return tu,ts,tv
    else:
        raise Exception("Svd(UniTensor)","[ERROR] Svd can only accept UniTensor")



def Svd_truncate(a, keepdim=None):
    #v0.3 OK
    """
    The function performs the svd to input UniTensor, and truncate [truncate] dim from the smallest singular value to the tensor. The UniTensor should be rank-2. each bond's dim should be >=2.


    Args:
        a :
            UniTensor[untagged], rank-2, 1 inbond 1 outbond.

        keepdim:
            integer, the keeping dimension. When set, it will keep only the largest "keepdim" singular values and their corresponding eigenvectors.


    Return: u , s , vt
        u :
            UniTensor[untagged], rank-2, 1 inbond 1 outbond, the truncated unitary matrix with shape (a.shape()[0], truncate)

        s :
            UniTensor[untagged], rank-2, 1 inbond 1 outbond, the diagonal, truncated singular matrix with shape (truncate,truncate)

        vt:
            UniTensor[untagged], rank-2, 1 inbond 1 outbond, the transposed right unitary matrix with shape (truncate,a.shape()[1])


    Example:
    ::
        y = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(4)],rowrank=1)
        y.SetElem([1,1,0,1,
                   0,0,0,1,
                   1,1,0,0])

    >>> print(y)
    Tensor name:
    is_diag    : False
    tensor([[1., 1., 0., 1.],
            [0., 0., 0., 1.],
            [1., 1., 0., 0.]], dtype=torch.float64)

    >>> u,s,vt = tor10.linalg.Svd_truncate(y,keepdim=2)
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

    >>> print(vt)
    Tensor name:
    is_diag    : False
    tensor([[-0.6280, -0.6280,  0.0000, -0.4597],
            [ 0.3251,  0.3251,  0.0000, -0.8881]], dtype=torch.float64)

    """
    if isinstance(a,UniTensor):

        ## Qnum_ipoint
        if a.is_symm:
            raise Exception("Qdr(a)","[Abort] curretly don't support symm tensor.")

        if a.is_diag:
            raise Exception("svd(a)","[Abort] svd currently don't support diagonal tensor.")

        if a.braket is not None:
            raise Exception("svd(a)","svd can only accept untagged UniTensor[regular]")

        if a.rowrank != 1:
            raise Exception("svd(a)","should be a UniTensor with 1 in-bond, 1 out-bond")

        u, s, v = torch.svd(a.Storage,some=True)

        tmp = np.argwhere(a.labels<0)
        if len(tmp) == 0:
            tmp = 0
        else:
            tmp = np.min(tmp)

        if keepdim is not None:
            if keepdim < 0 or keepdim > len(s):
                raise ValueError("Svd_truncate", "[ERROR] the keepdim=%d is invalid, must larger than 0 and smaller than the total number of eigenvalues." % keepdim)
            u = u[:, :keepdim]
            s = s[:keepdim]
            v = v[:, :keepdim]

        tu = UniTensor(bonds =[Bond(u.shape[0]),Bond(u.shape[1])],\
                      rowrank=1,\
                      labels=[a.labels[0],tmp-1],\
                      check=False)
        tu._mac(torch_tensor=u)

        tv = UniTensor(bonds =[Bond(v.shape[1]),Bond(v.shape[0])],\
                      rowrank=1,\
                      labels=[tmp-2,a.labels[1]],\
                      check=False)
        tv._mac(torch_tensor=v.transpose(0,1))

        ts = UniTensor(bonds  =[tu.bonds[1],tv.bonds[0]],\
                      labels =[tu.labels[1],tv.labels[0]],\
                      rowrank=1,\
                      check=False,\
                      is_diag=True)
        ts._mac(torch_tensor=s)

        return tu,ts,tv
    else:
        raise Exception("Svd_truncate(UniTensor,int)","[ERROR] Svd_truncate can only accept UniTensor")

def Matmul(a,b):
    """
    Performs matrix multiplication on the rank-2 UniTensors.

        :math:`A \cdot B`

    Note that both the UniTensors should be rank-2,untagged, and dimension should be matched.

    If a and b are both diagonal matrix, the return will be a diagonal matrix. If one (or both) of them are non-diagonal matrix and the other is diagonal matrix, the return will be a dense matrix.

    Args:
        a:
            The UniTensors that will be matrix-multiply, UniTensor should be [untagged]

        b:
            The UniTensors that will be matrix-multiply, UniTensor should be [untagged]

    Return:
        UniTensor,rank-2 tensor with 1 inbond 1 outbond.

    """
    if isinstance(a,UniTensor) and isinstance(b,UniTensor):

        ## [Note] no need to check if a,b are both rank 2. Rely on torch to do error handling!
       

        ## Qnum_ipoint
        if a.is_symm or b.is_symm :
            raise Exception("Matmul(a,b)","[Abort] Matmul cannot operate on sym TN.")

        if a.braket is not None or b.braket is not None:
            raise Exception("Matmul(a,b)","Matmul can only accept two regular Tensors")

        
        if a.rowrank != 1 or b.rowrank != 1:
            raise Exception("Matmul(a,b)","Matmul can only accept two UniTensor with each has 1 inbond and 1 outbond")

        if a.is_diag == b.is_diag:
            tmp = UniTensor(bonds =[a.bonds[0],b.bonds[1]],\
                            rowrank=1,\
                            check=False,\
                            is_diag=a.is_diag)
            tmp._mac(torch_tensor=torch.matmul(a.Storage,b.Storage))
        else:
            if a.is_diag:
                tmp = UniTensor(bonds =[a.bonds[0],b.bonds[1]],\
                                rowrank=1,\
                                check=False)
                tmp._mac(torch_tensor=torch.matmul(torch.diag(a.Storage),b.Storage))
            if b.is_diag:
                tmp = UniTensor(bonds =[a.bonds[0],b.bonds[1]],\
                                rowrank=1,\
                                check=False)
                tmp._mac(torch_tensor = torch.matmul(a.Storage,torch.diag(b.Storage)))
        return tmp

    else:
        raise TypeError("_Matmul(a,b)", "[ERROR] _Matmul can only accept UniTensors for both a & b")


def Chain_matmul(*args):
    """
    Performs matrix multiplication on all the UniTensors.

        :math:`A \cdot B \cdot C \cdot D \cdots`

    Note that

    1. all the UniTensors should be rank-2,untagged, and dimension should be matched.

    2. The input UniTensors can have some of them are diagonal matrix (is_diag=True). The return will always be a rank-2 UniTensor with is_diag=False

    Args:
        *args:
            The UniTensors that will be matrix-multiply. Each UniTensor should be [untagged] and with 1 inbond and 1 outbond

    Return:
        UniTensor,rank-2 tensor with 1 inbond, 1 outbond, and default labels.

    Example:
    ::
        a = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(4)],rowrank=1)
        b = tor10.UniTensor(bonds=[tor10.Bond(4),tor10.Bond(5)],rowrank=1)
        c = tor10.UniTensor(bonds=[tor10.Bond(5),tor10.Bond(6)],rowrank=1)
        d = tor10.UniTensor(bonds=[tor10.Bond(6),tor10.Bond(2)],rowrank=1)

    >>> f = tor10.Chain_matmul(a,b,c,d)
    >>> f.Print_diagram()
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

    """
    f = lambda x,idiag: torch.diag(x) if idiag else x
    isUT = all( isinstance(UT,UniTensor) for UT in args)


    ## Checking performance:
    #"""
    #for i in range(len(tmp_args)):
    #    if not tmp_args[i] is args[i].Storage:
    #       print("Fatal performance")
    #       exit(1)
    #"""

    if isUT:
        ## Qnum_ipoint
        if not all( UT.is_symm==False for UT in args):
            raise Exception("Chain_matmul(*args)","[Abort] Chain multiplication for symm tensor(s) are under developing.")

        if not all( UT.braket is None and UT.rowrank==1 for UT in args):
            raise Exception("Chain_matmul(*args)","Chain mult should have all UniTensor have 1 inbond 1 outbond")

        tmp_args = [f(args[i].Storage,args[i].is_diag) for i in range(len(args))]

        tmp = UniTensor(bonds =[args[0].bonds[0],args[-1].bonds[1]],\
                         rowrank=1,\
                         check=False)
        tmp._mac(torch_tensor = torch.chain_matmul(*tmp_args))
        return tmp

    else:
        raise TypeError("_Chain_matmul(*args)", "[ERROR] _Chain_matmul can only accept UniTensors for all elements in args")






def Inverse(a):
    #v0.3 OK
    """
    This function returns the inverse of a rank-2 tensor (matrix).

        :math:`a^{-1}`

    If the input UniTensor is diagonal, the return will also be a diagonal matrix.

    Args:
        a :
            A rank-2 UniTensor[untagged] (matrix), with 1-inbond. 1-outbond. Note that if the matrix is not inversable, error will be issued. passing a non-rank2 UniTensor, error will be issued.

    Return:
        UniTensor

    """
    if isinstance(a,UniTensor):
        if a.is_symm :
            raise TypeError("Inverse","[ERROR] cannot inverse a symmetry tensor")

        if a.braket is not None:
            raise TypeError("Inverse","[ERROR] inverse can only accept untagged UniTensor[regular]")

        if a.rowrank != 1:
            raise Exception("Inverse","[ERROR] inverse should have UniTensor with rowrank=1")

        if a.is_diag:
            a_inv = UniTensor(bonds = a.bonds,\
                          labels=a.labels,\
                          rowrank=1,\
                          is_diag=True,\
                          check=False)
            a_inv._mac(torch_tensor = a.Storage**-1)

        else:
            a_inv = UniTensor(bonds = a.bonds,\
                              labels=a.labels,\
                              rowrank=1,\
                              check=False)
            a_inv._mac(torch_tensor=torch.inverse(a.Storage))
        return a_inv
    else:
        raise Exception("Inverse(UniTensor)","[ERROR] Inverse can only accept UniTensor")


def Det(a): 
    #v0.3 OK
    """
    This function returns the determinant a rank-2 tensor.

    :math:`\det(a)`

    Args:
        a :
            a rank-2 UniTensor [untagged] (matrix) with 1 inbond 1 outbond.
    Return:
        UniTensor, 0-rank (constant)

    Example:
    ::
        a = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(3)])
        a.SetElem([4,-3,0,
                   2,-1,2,
                   1, 5,7])
        b = tor10.UniTensor(bonds=[tor10.Bond(3),tor10.Bond(3)],is_diag=True)
        b.SetElem([1,2,3])

    >>> print(a)
    Tensor name:
    is_diag    : False
    tensor([[ 4., -3.,  0.],
            [ 2., -1.,  2.],
            [ 1.,  5.,  7.]], dtype=torch.float64)

    >>> out = tor10.Det(a)
    >>> print(out)
    Tensor name:
    is_diag    : False
    tensor(-32., dtype=torch.float64)

    >>> print(b)
    Tensor name:
    is_diag    : True
    tensor([1., 2., 3.], dtype=torch.float64)

    >>> out = tor10.Det(b)
    >>> print(out)
    Tensor name:
    is_diag    : False
    tensor(6., dtype=torch.float64)

    """
    if isinstance(a,UniTensor):
        if a.is_symm:
            raise Exception("Det","[ERROR] cannot operate deteminant on a symmetry tensor")
        
        if a.braket is not None:
            raise Exception("Det","[ERROR] det can only operate on untagged UniTensor[regular]")
    
        if a.rowrank != 1:
            raise Exception("Det","[ERROR] det should have a rank-2 UniTensor with rowrank=1")

        if a.is_diag:
            tmp = torch.prod(a.Storage)
        else:
            tmp = torch.det(a.Storage)

        out =  UniTensor(bonds=[],labels=[],rowrank=0,check=False)
        out._mac(torch_tensor=tmp)
        return out

    else:
        raise Exception("Det(UniTensor)","[ERROR] Det can only accept UniTensor")

def Norm(a):
    """
    Returns the matrix norm of the UniTensor. The input tensor should be untagged. 

    If the given UniTensor is a matrix (rank-2), matrix norm will be calculated. If the given UniTensor is a vector (rank-1), vector norm will be calculated. If the given UniTensor has more than 2 ranks, the vector norm will be appllied to last dimension. 


    Args:
        a :
            a UniTensor[untagged]

    Return:
        UniTensor, 0-rank (constant)

    """

    if isinstance(a,UniTensor):
        if a.is_symm :
            raise Exception("Norm","[ERROR] cannot operate Norm on a symmetry tensor")

        if a.braket is not None:
            raise Exception("Norm","[ERROR] Norm can only operate on untagged UniTensor")

        if a.rowrank != 1 or len(a.labels)!=2:
            raise Exception("Norm","[ERROR] the input UniTensor should be rank-2, untagged with rowrank=1")


        #tmp = torch.norm(a.Storage)
        #if len(tmp.shape) != 0:
        #    return UniTensor(bonds=[tor10.Bond(tmp.shape[i]) for i in range(len(tmp.shape))],rowrank=1,torch_tensor=tmp,check=False)
        #else:
        tmp = UniTensor(bonds=[],labels=[],rowrank=0,check=False)
        tmp._mac(torch_tensor = torch.norm(a.Storage))
        return tmp
    else:
        raise Exception("Norm(UniTensor)","[ERROR] Norm can only accept UniTensor")
