![alt text](./Tor10_icon.png)

## Requirements
    pytorch>=1.0
    numpy>=1.15

## Code naming principle:
    1) the functions start with "_" are the private function that should not be call directly by user.

## Feature:
        
    1. Create Tensor:
        * support multiple precisions.        
        * support devices (cpu and gpu are trivial)
        * preserve the similar api for Bond 
        
        ```
        bds = [ Bond(BD_IN,3), Bond(BD_OUT,4)]
        A = UniTensor(bds,label=[2,4],dtype=torch.float64,device=torch.device("cpu"))
        ```
    2. Tensor :
        * vitual swap and reshape/permute are avaliable implicitly.
        * Use Contiguous() when needed.
        ```
        A.Contiguous()
        ```
    3. Multiple Symmetries:
        * Support arbitrary number of symmetry. 
        * see test_sym.py for how to use them. 
        

    See test.py for further detail application functions.

## Example:

    See iTEBD.py for an simple example of using iTEBD algo. to calculate the 1D-transverse field Ising model 
    See iTEBD_gpu.py for an simple example of the same algo accelerated with GPU. 



## Developers:

    * Kai-Hsin Wu     kaihsinwu@gmail.com

    * Jing-Jer Yen 
    * Yen-Hsin Wu 
