![alt text](./Tor10_icon.png)

## Requirements
    pytorch>=1.0
    numpy  >=1.15
    sphinx >=1.8.2

## Code naming principle:
    1) the functions start with "_" are the private function that should not be call directly by user.

## Feature:
        
    1. Create Tensor:
        * support multiple precisions.        
        * support devices (cpu and gpu are trivial)
        * preserve the similar api for Bond 
        
```python
       ## create a rank-2 Tensor 
       bds = [ Bond(BD_IN,3), Bond(BD_OUT,4)]
       A = UniTensor(bds,label=[2,4],dtype=torch.float64,device=torch.device("cpu"))

       ## Moving to GPU:
       A.to(torch.device("cuda:0"))
```

    2. Tensor :
        * vitual swap and permute. All the permute and swap will not change the underlying memory
        * Use Contiguous() when needed to actual moving the memory layout.

```python
        A.Contiguous()
```

    3. Multiple Symmetries:
        * Support arbitrary number of symmetry. 
        * see test_sym.py for how to use them. 
        
    4. Network :
        * See test_ntwrk.py for how to use network.
        * See test.net for how to defined a Network file.



    See test.py for further detail application functions.

## Example:

    See iTEBD.py for an simple example of using iTEBD algo. to calculate the 1D-transverse field Ising model 
    See iTEBD_gpu.py for an simple example of the same algo accelerated with GPU. 

## Documentation:

[https://kaihsinwu.gitlab.io/Tor10/](https://kaihsinwu.gitlab.io/Tor10/)

## Developers:

    * Kai-Hsin Wu     kaihsinwu@gmail.com

    * Jing-Jer Yen 
    * Yen-Hsin Wu 
