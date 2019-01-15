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
        * preserve the same api for Bond (currently no Qnum and Symm)
        
        bds = [ Bond(BD_IN,3), Bond(BD_OUT,4)]
        A = UniTensor(bds,label=[2,4],dtype=torch.float64,device=torch.device("cpu"))

    2. Tensor :
        * vitual swap and reshape/permute are avaliable implicitly.
        * Use Contiguous() when needed.

        A.Contiguous()

        

    See test.py for further detail application functions.

## Example:

    See iTEBD.py for an simple example of using iTEBD algo. to calculate the 1D-transverse field Ising model 
    See iTEBD_gpu.py for an simple example of the same algo accelerated with GPU. 


## Note:
    
    1. UniTensor: 
        a. When created, regardless of the Bond sequences in the Bond list that pass into the argument, All the IN-bond will be force to the smaller index. 
            Ex: bonds = [Bond(BD_IN,3), Bond(BD_OUT,5),Bond(BD_IN,2),Bond(BD_OUT,4)]
                the UniTensor instance that created will have shape (3,2,5,4); 
                with enforeced property bonds=[Bond(BD_IN,3), Bond(BD_IN,2), Bond(BD_OUT,5),Bond(BD_OUT,4)]


## Developers:

    * Kai-Hsin Wu     kaihsinwu@gmail.com
    * Jing-Jer Yen 
    * Yen-Hsin Wu 
