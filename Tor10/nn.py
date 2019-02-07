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
                self.P1 = Tor10.nn.Parameter(Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,2),Tor10.Bond(Tor10.BD_OUT,2)]))
                self.P2 = Tor10.nn.Parameter(Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,2),Tor10.Bond(Tor10.BD_OUT,2)]))
 
            def forward(self,x):
                y = Tor10.Matmul(Tor10.Matmul(x,self.P1),self.P2)
                return y

    >>> x = Tor10.UniTensor(bonds=[Tor10.Bond(Tor10.BD_IN,2),Tor10.Bond(Tor10.BD_OUT,2)])
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
            while(1):
                if hasattr(instance,'param_%d'%(n)):
                    n+=1
                else:
                    setattr(instance,'param_%d'%(n),data.Storage)
                    break
 
    return data
        

