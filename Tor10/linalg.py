from .UniTensor import *
import torch 
import numpy as np



def ExpH(a):
    """
    This function performs the exp^{H} where H is the hermitian matrix. 
    The Intricate computation follows procedure: symeig() first and exp() the singular matrix.

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
        @description : The function performs the qr to input UniTensor [a]. The UniTensor should be rank-2 with 1-inbond 1-outbond. each inbond and outbond's dim should be >=2. 
                       Mathmatically, a = q \cdot r
        @params      :  a : UniTensor, rank-2, 1 inbond 1 outbond.
        @return      :  q , r  
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
        @description : The function performs the qr to input UniTensor [a]. The UniTensor should be rank-2 with 1-inbond 1-outbond. each inbond and outbond's dim should be >=2. 
                       Mathmatically, a = q \cdot r
        @params      :  a : UniTensor, rank-2, 1 inbond 1 outbond.
        @return      :  q , r  
                        q : UniTensor, 2-rank, 1 inbond 1 outbond, the unitary matrix
                        r : UniTensor, 2-rank, 1 inbond 1 outbond, the upper triangular matrix 
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
        @description : The function performs the svd to input UniTensor [a]. The UniTensor should be rank-2 with 1-inbond 1-outbond. each inbond and outbond's dim should be >=2. 
                       Mathmatically, a = u \cdot s \cdot vt
        @params      :  a : UniTensor, rank-2, 1 inbond 1 outbond.
        @return      :  u , s , vt 
                        u : UniTensor, 2-rank, 1 inbond 1 outbond, the unitary matrix
                        s : UniTensor, 2-rank, 1 inbond 1 outbond, the diagonal, singular matrix 
                        vt: UniTensor, 2-rank, 1 inbond 1 outbond, the transposed right unitary matrix
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
        @description : The function performs the svd to input UniTensor [a]. The UniTensor should be rank-2 with 1-inbond 1-outbond. each inbond and outbond's dim should be >=2. 
                       Mathmatically, a = u \cdot s \cdot vt
        @params      :  a : UniTensor, rank-2, 1 inbond 1 outbond.
        @return      :  u , s , vt 
                        u : UniTensor, 2-rank, 1 inbond 1 outbond, the unitary matrix
                        s : UniTensor, 2-rank, 1 inbond 1 outbond, the diagonal, singular matrix 
                        vt: UniTensor, 2-rank, 1 inbond 1 outbond, the transposed right unitary matrix
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
    
    if isinstance(a,UniTensor) and isinstance(b,UniTensor):

        ## [Note] no need to check if a,b are both rank 2. Rely on torch to do error handling! 

        ## Qnum_ipoint
        if a.bonds[0].qnums is not None or b.bonds[0].qnums is not None:
            raise Exception("Matmul(a,b)","[Abort] Matmul for sym TN is under developing.")

        if a.is_diag == b.is_diag:
            tmp = UniTensor(bonds =[a.bonds[0],b.bonds[1]],\
                            labels=[a.labels[0],b.labels[1]],\
                            torch_tensor=torch.matmul(a.Storage,b.Storage),\
                            check=False,\
                            is_diag=a.is_diag)
        else:
            if a.is_diag:
                tmp = UniTensor(bonds =[a.bonds[0],b.bonds[1]],\
                                labels=[a.labels[0],b.labels[1]],\
                                torch_tensor=torch.matmul(torch.diag(a.Storage),b.Storage),\
                                check=False)
            if b.is_diag:
                tmp = UniTensor(bonds =[a.bonds[0],b.bonds[1]],\
                                labels=[a.labels[0],b.labels[1]],\
                                torch_tensor=torch.matmul(a.Storage,torch.diag(b.Storage)),\
                                check=False)

        return tmp

    else:
        raise TypeError("_Matmul(a,b)", "[ERROR] _Matmul can only accept UniTensors for both a & b")


def Chain_matmul(*args):
    """
        @description: This function performs matrix multiplication on all the UniTensors. Note that all the UniTensors should be rank-2 with 1-inbond 1-outbond

        @params     : The UniTensors that will be matrix-multiply

        @return     : UniTensor,rank-2, 1 inbond 1 outbond. The label of inbond = the label of inbond of first UniTensor. The label of outbond = the label of outbond of the last UniTensor.
        @exampe     : 
                        f = Chain_matmul(a,b,c,d,e)
                        Mathmatically equivalent as : f = a \cdot b \cdot c \cdot d \cdot e

    """
    f = lambda x,idiag: torch.diag(x) if idiag else x 
    isUT = all( isinstance(UT,UniTensor) for UT in args)    
    
    
    tmp_args = [f(args[i].Storage,args[i].is_diag) for i in range(len(args))] 

    ## Checking performance:
    """  
    for i in range(len(tmp_args)):
        if not tmp_args[i] is args[i].Storage:
           print("Fatal performance")
           exit(1) 
    """

    if isUT:
        ## Qnum_ipoint
        if not all( (UT.bonds[0].qnums is None) for UT in args):
            raise Exception("Chain_matmul(*args)","[Abort] Chain multiplication for symm tensor(s) are under developing.")


        return UniTensor(bonds =[args[0].bonds[0],args[-1].bonds[1]],\
                         labels=[args[0].labels[0],args[-1].labels[1]],\
                         torch_tensor=torch.chain_matmul(*tmp_args),\
                         check=False)

    else:
        raise TypeError("_Chain_matmul(*args)", "[ERROR] _Chain_matmul can only accept UniTensors for all elements in args")






def Inverse(a):
    """
        @description: This function returns the inverse of a rank-2 tensor.
        @params     : 
                      a : UniTensor
        @return     : Unitensor
                    
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
        @description: This function returns the determinant a rank-2 tensor.
        @params     : 
                      a : rank-2 UniTensor
        @return     : a 0-dimension tensor contains the determinant of input
                    
    """
    if isinstance(a,UniTensor):

        if a.is_diag:
            return torch.prod(a.Storage)
        else:
            return torch.det(a.Storage)
    else:
        raise Exception("Det(UniTensor)","[ERROR] Det can only accept UniTensor")

def Norm(a):
    """
        @description: This function returns the frobinieus 2-norm of a tensor.
        @params     : 
                      a : UniTensor
        @return     : a 0-dimension tensor contains the 2-norm of input
                    
    """

    if isinstance(a,UniTensor):
        return torch.norm(a.Storage)
    else:
        raise Exception("Norm(UniTensor)","[ERROR] Norm can only accept UniTensor")
