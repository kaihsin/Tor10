import torch
import inspect
from .UniTensor import *

def Parameter(data,requires_grad=True):
    """
    Convert a UniTensor to be considered as a module parameter. 

    They have a special property when use with torch.nn.Module. When they are assigned as Module attributes, they are automatically added to the list of its parameter, and appears in Module.parameters. (This is similar as torch.nn.Parameter)

    Args:
        data: 
            UniTensor, parameter tensor.

        requires_grad:        
            bool, if the parameter requires gradient. 

    Return:
        UniTensor, with Paramter property.

    Example:
    ::
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model,self).__init__()
                ## Customize and register the parameter.
                self.P1 = tor10.nn.Parameter(tor10.UniTensor(bonds=[tor10.Bond(2),tor10.Bond(2)],rowrank=1))
                self.P2 = tor10.nn.Parameter(tor10.UniTensor(bonds=[tor10.Bond(2),tor10.Bond(2)],rowrank=1))
 
            def forward(self,x):
                y = tor10.Matmul(tor10.Matmul(x,self.P1),self.P2)
                return y

    >>> md = Model()
    >>> print(list(md.parameters()))
    [Parameter containing:
    tensor([[0., 0.],
            [0., 0.]], dtype=torch.float64, requires_grad=True), Parameter containing:
    tensor([[0., 0.],
            [0., 0.]], dtype=torch.float64, requires_grad=True)]
    
    """

    if not isinstance(data,UniTensor):
        raise TypeError("nn.Parameter","[ERROR] data should be an UniTensor")

    if data.braket is not None:
        raise TypeError("nn.Parameter","[ERROR] data can only be an untagged uniTensor")
    

    data.Storage = torch.nn.Parameter(data.Storage,requires_grad=requires_grad)


    ## Get the mother instance
    frame = inspect.stack()[1][0]
    args,_,_,value_dict = inspect.getargvalues(frame)
    if len(args) and args[0] == 'self':
        instance = value_dict.get('self',None)
    else:
        instance=None


    if instance is not None:
        #print("OK")
        if isinstance(instance,torch.nn.Module):
            #print("OK module")
            
            n=0
            while 1:
                if hasattr(instance,'param_%d' % n):
                    n+=1
                else:
                    setattr(instance,'param_%d' % n, data.Storage)
                    break
 
    return data
        

class Linear:
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Note that the the input and output UniTensors will have shape:

        - Input: :math:`(N, *, \text{in\_features})` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, \text{out\_features})` where all but the last dimension
          are the same shape as the input.


    Args:
        in_features: 
            uint, size of each input sample
        
        out_features: 
            uint, size of each output sample
        
        bias: 
            bool, If set to False, the layer will not learn an additive bias.


    Attributes:
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = tor10.nn.Linear(20, 30)
        >>> iput = tor10.From_torch(torch.randn(128, 20),rowrank=1)
        >>> oput = m(iput)
        >>> print(oput.shape)
        torch.Size([128, 30])
    
    """
    def __init__(self,in_features,out_features,bias=True):
        self.tnn = torch.nn.Linear(in_features,out_features,bias=bias)

        ## hook 
        ## Get the mother instance
        frame = inspect.stack()[1][0]
        args,_,_,value_dict = inspect.getargvalues(frame)
        if len(args) and args[0] == 'self':
            instance = value_dict.get('self',None)
        else:
            instance=None

        if instance is not None:
            #print("OK")
            if isinstance(instance,torch.nn.Module):
                #print("OK module")

                n=0
                while 1:
                    if hasattr(instance,'param_%d' % n):
                        n+=1
                    else:
                        setattr(instance,'param_%d' % n, self.tnn)
                        break

    def __call__(self,ipt):
        return self.forward(ipt)

    def forward(self,ipt):
        if not isinstance(ipt,UniTensor):
            raise TypeError("tor10.nn.Linear","[ERROR] can only accept UniTensor")
    
        out = self.tnn(ipt.Storage)
        
        tmp = UniTensor(bonds=np.append(copy.deepcopy(ipt.bonds[:-1]),Bond(self.tnn.out_features)),rowrank=len(ipt.bonds[:-1]),check=False)
        tmp._mac(torch_tensor = out)

        return tmp
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(\
            self.tnn.in_features, self.tnn.out_features, self.tnn.bias is not None)

    def weight(self):
        """
        Return the learnable weights of the module of shape
        :math:`(\text{out\_features}, \text{in\_features})`. The values are
        initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
        :math:`k = \frac{1}{\text{in\_features}}`

        Return:
            UniTensor, rank-2
        """
        tmp = UniTensor(bonds=[Bond(self.tnn.out_features),Bond(self.tnn.in_features)],rowrank=1,check=False)
        tmp._mac(torch_tensor=self.tnn.weight)
        return tmp

    def bias(self):
        """
        the learnable bias of the module of shape :math:`(\text{out\_features})`.
        If :attr:`bias` is ``True``, the values are initialized from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
        :math:`k = \frac{1}{\text{in\_features}}`

        Return:
            if :attr:`bias`==True, return a UniTensor of bias; if False, return None
                
        """
        if self.tnn.bias is None:
            return None
        else:
            tmp = UniTensor(bonds=[Bond(self.bias.shape[0])],rowrank=0,check=False)
            tmp._mac(torch_tensor=self.tnn.bias)
            return tmp

