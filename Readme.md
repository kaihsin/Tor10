![alt text](./Tor10_icon.png)

## What's new
    v0.2.1
    1. Fix bugs for UniTensor.shape() if is_diag = True
    2. Change the Definition of Bond. The In-bond and Out-bond are defined within the UniTensor. (see documentation for details)
    3. Add new sparse structure: [is_blockform] to efficiently store the tensor with symmetry. (currently has limited function support.)
    4. Add U1 and Zn Symmetry class. This object as the generator that handle the rule for combine bonds and Symmetry stuff in the UniTensor (for future develope
    5. Update the example.py
    6. Change shape to @property
    7. The Permute and Contract can only be intra- in-bond/out-bond.
    8. in aspect of 7., Permute API is changed.     
    9. Reshape will return new UniTensor, as it re-define the underlying physical space. 
    10. Explicity define the in-bond as bra, out-bond as ket
    11. Hosvd, the return Tensor in/out bond layout
    12. Add UniTensor.Whole_transpose(), it is use to permute all in-bond to out-bond and vise versa.    

    v0.2.2
    1. Add SetName() 
    2. Fix SetElem() input bug. The correct functionality is can only accept 1D list or 1D ndarray
    3. Add item() to get python scalar from a one element UniTensor
    4. __getitem__ now return a UniTensor, not a torch tensor.  
    5. Fix from_torch bug for a 0-rank torch tensor.
    6. Fix Svd bug. This bug is due to accidentally insert "OUT" in the code when editing.
    7. Changed Svd return behavior.The return u,s,v UniTensors' in/out bond layout will be: s (2in2out); u(1out+the original first bond's type of input tensor); v(1in+the original second bond's type of input tensor)

    v0.3
    1. Three type of UniTensor can be construct. [symmetry][tagged][untagged] 
    2. The Bond can be tagged with bra/ket or untagged (regular)
    3. Now change the storage of symmetry to blockform. the dense storage for symmetry is deprecated.
    4. Add braketform() to coalesce the bra/ket with row/col(in/out)
    5. N_rowrank defines the row/col space. 
    6. Enhanced Print_diagram
    7. Seperate Contiguous() and Contiguous\_() for inplace and return type function.
    8. change behavior of GetBlock and PutBlock on a non-symm tensor. in this version, it will return a rank-2 matrix.
    9. Add GetValidQnums() for symmetry tensor
    10. For symmtry, the bond order is relevant. The default fusion tree is in order of ((((0,1),2),3),4 ....
    11. Add UniTensor.dtype() [@property]
    12. Add UniTensor.device() [@property]

## Release version
    v0.2.2 -> v0.3 deving

## Requirements
    pytorch>=1.0
    numpy  >=1.15
    sphinx >=1.8.2
    sphinx_rtd_theme >=0.4.2 

## Documentation:

[https://kaihsinwu.gitlab.io/tor10](https://kaihsinwu.gitlab.io/tor10)

## Code naming principle:
    1) the functions start with "_" are the private function that should not be call directly by user.

## Feature:
        
    1. Create Tensor:
        * support multiple precisions.        
        * support devices (cpu and gpu are trivial)
        * preserve the similar api for Bond 
        
```python
       ## create a rank-2 Tensor 
       bds = [ Bond(3,BD_BRA), Bond(4,BD_KET)]
       A = UniTensor(bds,label=[2,4],N_inbond=1,dtype=torch.float64,device=torch.device("cpu"))

       ## Moving to GPU:
       A.to(torch.device("cuda:0"))
```

    2. Tensor :
        * vitual swap and permute. All the permute and swap will not change the underlying memory
        * Use Contiguous() or Contiguous_() when needed to actual moving the memory layout.

```python
        A.Contiguous()
```

    3. Multiple Symmetries:
        * Support arbitrary numbers and types of symmetry.
        * Currently support U1 and Zn (with arbitrary n). 

```python
        #> Multiple mix symmetry: U1 x Z2 x Z4
        bd_sym_mix = Tor10.Bond(3,qnums=[[-2,0,0],
                                         [-1,1,3],
                                         [ 1,0,2]],
                                 sym_types=[Tor10.Symmetry.U1(),
                                            Tor10.Symmetry.Zn(2),
                                            Tor10.Symmetry.Zn(4)],
                                  Tor10.BD_BRA)
``` 
        
    4. Network :
        * See documentation for how to use network.

    5. Autograd mechanism:
        The Tor10 now support the autograd functionality. The Contract, Matmul etc will automatically contruct the gradient flow for UniTensor that has [requires_grad=True]
        
        * See documentation for further details


    6. Easy coordinate with pytorch for Neural-Network:
        We provide Tor10.nn that can easy cooperate with pytorch.nn.Module to perform neural-network tasks.

```python
        import torch
        import Tor10
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model,self).__init__()
                ## Customize and register the parameter.
                self.P1 = Tor10.nn.Parameter(Tor10.UniTensor(bonds=[Tor10.Bond(2,BD_BRA),Tor10.Bond(2,BD_KET)]))
                self.P2 = Tor10.nn.Parameter(Tor10.UniTensor(bonds=[Tor10.Bond(2,BD_BRA),Tor10.Bond(2,BD_KET)]))
 
            def forward(self,x):
                y = Tor10.Matmul(Tor10.Matmul(x,self.P1),self.P2)
                return y

        md = Model()
        print(list(md.parameters()))
        ## Output:
        #    [Parameter containing:
        #    tensor([[0., 0.],
        #            [0., 0.]], dtype=torch.float64, requires_grad=True), Parameter containing:
        #    tensor([[0., 0.],
        #            [0., 0.]], dtype=torch.float64, requires_grad=True)]
```
        * See documentation for further details


    See test.py for further detail application functions.

## Example:

    See iTEBD.py for an simple example of using iTEBD algo. to calculate the 1D-transverse field Ising model 
    See iTEBD_gpu.py for an simple example of the same algo accelerated with GPU. 

    See example.py for elementary usage.

## Developers:

    * Kai-Hsin Wu     kaihsinwu@gmail.com

    * Jing-Jer Yen 
    * Yen-Hsin Wu 
