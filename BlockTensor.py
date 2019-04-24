import torch as tor
import numpy as np 
import copy 

def combine(ins,braket):
    ## if corresponding elem in braket is 1, then all qnums will be mulipy negative
    sign = braket
    #print(sign)
    if len(ins) < 2:
        return ins
    else:   
        out = (ins[0]*sign[0]).reshape(-1,1) + (ins[1]*sign[1]).reshape(1,-1)
        out = out.flatten()
        for i in np.arange(2,len(ins),1):
            out = out.reshape(-1,1) + (ins[i]*sign[i]).reshape(1,-1)
            out = out.flatten()
        return out

def decompress_idx(x,accu_offsets):

    y = []
    for i in range(len(accu_offsets)):
        y.append(np.array(x/accu_offsets[i]).astype(np.int))
        x = x%accu_offsets[i]
    return np.array(y).swapaxes(0,1)


bq1 = np.array([0,-1,-1,-2,0,2,0,1])
bq2 = np.array([0, 2, 2,-2,1,0])
bq3 = np.array([2, -2,1,-1,0])
bq4 = np.array([-2,-2,1,1])

class BlockTensor:
    def __init__(self,bqs, N_inbond,braket=None):
        self.bqs = np.array(copy.copy(bqs))
        self.braket = braket
        if self.braket is None:
            self.braket = np.array([1 if i<N_inbond else -1 for i in range(len(bqs))])
            self._N_rowrank = N_inbond
        else:
            if len(braket) != len(bqs):
                raise Exception("[ERROR] braket mush have the same len as # of bonds")
            self.braket = np.array(braket).astype(np.int)
            self._N_rowrank = len(np.argwhere(self.braket==1).flatten())

        self._rank = len(bqs)
        self._contiguous = True
        self.N_inbond = N_inbond

        ## check if it is in braket form.
        self._ibraket = False     
        if (self.braket[:N_inbond]==1).all() and (self.braket[N_inbond:]==-1).all():
            self._ibraket = True


        ## these three are the property that track the contiguous
        self._mapper = np.arange(len(bqs)).astype(np.int)
        self._inv_mapper = copy.copy(self._mapper)


        ## Calculate vaild qnums
        bq_in = combine(bqs[:N_inbond],self.braket[:N_inbond])
        uni_qnums_in = np.unique(bq_in)
        bq_out = combine(bqs[N_inbond:],self.braket[N_inbond:]*-1)
        uni_qnums_out = np.unique(bq_out)
        self._total_qnums = np.intersect1d(uni_qnums_in,uni_qnums_out)
        self._dict_qnum2blkid = dict(zip(self._total_qnums,range(len(self._total_qnums))))
        print(self._total_qnums)        

        ##accumulate offsets:
        self.accu_off = []
        tmp = 1
        for i in range(self._rank):
            self.accu_off.append(tmp)
            tmp*= len(self.bqs[-1-i])
        self.accu_off = np.array(self.accu_off[::-1])
        print(self.accu_off)
        #self._accu_off_in = accu_off[:N_inbond]/accu_off[N_inbond-1]
        #self._accu_off_out= accu_off[N_inbond:]

        ## calculate in mapper
        whe = [np.argwhere(bq_in==q).flatten() for q in self._total_qnums]
        self._in_invmapper_blks = -np.ones((len(bq_in),2)).astype(np.int)
        for b in range(len(whe)):
            self._in_invmapper_blks[whe[b],0] = b
            self._in_invmapper_blks[whe[b],1] = np.arange(len(whe[b])).astype(np.int)


        Nblocks = len(whe)
        self._in_mapper_blks = [ decompress_idx(whe[x],self.accu_off[:N_inbond]/self.accu_off[N_inbond-1]) for x in range(Nblocks)]

        ## calculate out mapper
        whe = [ np.argwhere(bq_out==q).flatten() for q in self._total_qnums]
        self._out_invmapper_blks = -np.ones((len(bq_out),2)).astype(np.int)
        for b in range(len(whe)):
            self._out_invmapper_blks[whe[b],0] = b
            self._out_invmapper_blks[whe[b],1] = np.arange(len(whe[b])).astype(np.int)

        self._out_mapper_blks = [ decompress_idx(whe[x],self.accu_off[N_inbond:]) for x in range(Nblocks)]



        self.Blocks = [tor.zeros((len(self._in_mapper_blks[iq]),len(self._out_mapper_blks[iq]))).to(tor.float64) for iq in range(Nblocks)]

        print(self.Blocks)

    def permute(self,*ids,N_inbond=None):
        if len(ids) != len(self._mapper):
            raise Exception("ERROR")
        print(ids)
        ids = list(ids)
        if N_inbond is not None:
            if N_inbond < 1:
                raise Exception("[ERROR]")
            self.N_inbond = N_inbond

        self._mapper = self._mapper[ids]
        self.braket = self.braket[ids]
        self.bqs = self.bqs[ids]
        Arr_range = np.arange(len(self._mapper))

        if (self._mapper == Arr_range).all():
            self._contiguous = True
        else:
            self._contiguous = False

        print(self.braket)
        if (self.braket[:self.N_inbond]==1).all() and (self.braket[self.N_inbond:]==-1).all():
            self._ibraket = True
        else:
            self._ibraket = False

        self._inv_mapper = np.zeros(len(self._mapper))
        self._inv_mapper[self._mapper] = Arr_range
        self._inv_mapper = self._inv_mapper.astype(np.int)
        print(self._mapper)
        print(self._inv_mapper)
        
        


    @property
    def is_contiguous(self):
        return self._contiguous

    @property
    def is_braket(self):
        return self._ibraket


    def to_braket_form(self):
        if self._ibraket:
            return self
        else:
            tb_mapper = np.argsort(self.braket)[::-1]
            rollback_mapper = np.zeros(len(tb_mapper)).astype(np.int)
            rollback_mapper[tb_mapper] = np.array(range(len(tb_mapper))).astype(np.int)
            
            self.permute(*tb_mapper)
            out = self.contiguous() ## create a new copy.
            self.permute(*rollback_mapper)        
            return out
            
    def contiguous(self):
        if self._contiguous:
            return self
        else:
            out = BlockTensor(self.bqs,N_inbond=self.N_inbond,braket=self.braket)
            out_accu_off_in = (out.accu_off[:out.N_inbond]/out.accu_off[out.N_inbond-1]).astype(np.int)
            out_accu_off_out= out.accu_off[out.N_inbond:]

            ## copy elements:
            for b in range(len(self.Blocks)):
                oldshape = self.Blocks[b].shape
                for i in range(oldshape[0]):
                    for j in range(oldshape[1]):
                        oldidx = np.concatenate((self._in_mapper_blks[b][i],self._out_mapper_blks[b][j]))
                        newidx = oldidx[self._mapper]
                        #print(oldidx,newidx)
                        new_row = int(np.sum(out_accu_off_in*newidx[:out.N_inbond]))
                        new_col = int(np.sum(out_accu_off_out*newidx[out.N_inbond:]))
                        b_id_in = out._in_invmapper_blks[new_row]
                        b_id_out = out._out_invmapper_blks[new_col]
                        ## check
                        if b_id_in[0] < 0 or b_id_out[0]<0:
                            raise Exception("[ERROR]")
                        if b_id_in[0] != b_id_out[0]:
                            print(b_id_in[0],b_id_out[0])
                            print("[ERROR!]")
                            exit(1)

                        out.Blocks[b_id_in[0]][b_id_in[1],b_id_out[1]] = self.Blocks[b][i,j]
 
            return out    


        
        

"""
##in-bond
print(bq1,bq2)
## out-bond
print(bq3,bq4)
bq_out = combine(bq3,bq4)
uni_qnums_out = np.unique(bq_out)

whe = [np.argwhere(bq_out==q).flatten() for q in uni_qnums_out]
Nblocks = len(whe)
r_mapper_blks = [ np.array([ (whe[x]/len(bq4)).astype(np.int), (whe[x]%len(bq4)).astype(np.int) ]) for x in range(Nblocks)]

######################
print(l_mapper_blks)
print(r_mapper_blks)
"""
bqs = [bq1,bq2,bq3,bq4]
a = BlockTensor(bqs,2)
a.permute(1,3,0,2)
print(a.is_contiguous,a.is_braket)
b = a.contiguous()
print(b.is_contiguous,b.is_braket)
d = b.to_braket_form()
print(d.is_contiguous,d.is_braket)

#a.permute(1,0,2,3)
#print(a.is_contiguous)


