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
## Developers:

    * Kai-Hsin Wu     kaihsinwu@gmail.com
    * Jing-Jer Yen 
    * Yen-Hsin Wu 
