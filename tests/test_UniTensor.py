import unittest
from context import tor10
import numpy as np
class TestUniTensorObjects(unittest.TestCase):

#    def setUp(self):
#        self.bds_x = [tor10.Bond(5), tor10.Bond(5), tor10.Bond(3)]
#        self.x = tor10.UniTensor(bonds=self.bds_x, N_inbond=1, labels=[4, 3, 5])

    def test_SetLabel(self):
        bds_x = [tor10.Bond(5), tor10.Bond(5), tor10.Bond(3)]
        x = tor10.UniTensor(bonds=bds_x, rowrank=1, labels=[4, 3, 5])
        x.SetLabel(-1, 2)
        self.assertEqual(x.labels[2],-1)

    def test_SetLabels(self):
        bds_x = [tor10.Bond(5), tor10.Bond(5), tor10.Bond(3)]
        x = tor10.UniTensor(bonds=bds_x, rowrank=1, labels=[4, 3, 5])
        x.SetLabels([3,2,1])
        self.assertListEqual(list(x.labels),[3,2,1])

    def test_Reshape(self):
        bds_x = [tor10.Bond(6), tor10.Bond(5), tor10.Bond(3)]
        x = tor10.UniTensor(bonds=bds_x, rowrank=1, labels=[4, 3, 5])
        y=x.Reshape([2, 3, 5, 3], new_labels=[1, 2, 3, -1], rowrank=2)
        self.assertListEqual(list(y.labels),[1, 2, 3, -1])

    def test_Reshape_(self):
        bds_x = [tor10.Bond(6), tor10.Bond(5), tor10.Bond(3)]
        x = tor10.UniTensor(bonds=bds_x, rowrank=1, labels=[4, 3, 5])
        x.Reshape_([2, 3, 5, 3], new_labels=[1, 2, 3, -1], rowrank=2)
        self.assertListEqual(list(x.labels), [1, 2, 3, -1])

    def test_CombineBonds(self):
        bds_x = [tor10.Bond(5), tor10.Bond(5), tor10.Bond(3)]
        x = tor10.UniTensor(bonds=bds_x, rowrank=2, labels=[4, 3, 5])
        x.CombineBonds([5, 3])
        self.assertListEqual(list(x.shape),[5,15])
        self.assertListEqual(list(x.labels), [4,5])

        y = tor10.UniTensor(bonds=bds_x, rowrank=2, labels=[4, 3, 5])
        y.CombineBonds([3,5])
        self.assertListEqual(list(y.shape), [15, 5])
        self.assertListEqual(list(y.labels), [3, 4])

        z = tor10.UniTensor(bonds=bds_x * 2, rowrank=3, labels=[4, 3, 5, 6, 7, 8])
        z.CombineBonds([4, 5, 6])
        self.assertListEqual(list(z.shape), [75, 5, 5, 3])
        self.assertListEqual(list(z.labels), [4, 3, 7, 8])
        self.assertEqual(z.rowrank, 1)

        z2 = tor10.UniTensor(bonds=bds_x * 2, rowrank=3, labels=[4, 3, 5, 6, 7, 8])
        z2.CombineBonds([4,5,6],permute_back=True)
        self.assertListEqual(list(z2.shape), [75, 5, 5, 3])
        self.assertListEqual(list(z2.labels), [4, 3, 7, 8])
        self.assertEqual(z2.rowrank, 2)

    def test_Permute(self):
        bds_x = [tor10.Bond(6), tor10.Bond(5), tor10.Bond(4), tor10.Bond(3), tor10.Bond(2)]
        x = tor10.UniTensor(bonds=bds_x, rowrank=3, labels=[1, 3, 5, 7, 8])
        x.Permute([0,2,1,4,3])
        self.assertListEqual(list(x.labels),[1, 5, 3, 8, 7])
        self.assertEqual(x.rowrank, 3)
        self.assertListEqual(list(x.shape),[6, 4, 5, 2 ,3])
        self.assertFalse(x.is_contiguous())


        y=tor10.UniTensor(bonds=bds_x, rowrank=3, labels=[1, 3, 5, 7, 8])
        y.Permute([3,1,5,7,8],by_label=True)
        self.assertListEqual(list(y.labels),[3, 1, 5, 7, 8])
        self.assertEqual(y.rowrank, 3)
        self.assertListEqual(list(y.shape), [5, 6, 4, 3, 2])
        self.assertFalse(y.is_contiguous())


        z=tor10.UniTensor(bonds=bds_x, rowrank=3, labels=[1, 3, 5, 7, 8])
        z.Permute([3, 1, 5, 7, 8], rowrank=2, by_label=True)
        self.assertListEqual(list(z.labels),[3, 1, 5, 7, 8])
        self.assertEqual(z.rowrank,2)
        self.assertListEqual(list(z.shape), [5, 6, 4, 3, 2])
        self.assertFalse(z.is_contiguous())

    def test_View(self):
        pass

    def test_GetTotalQnums(self):
        bd_sym_1 = tor10.Bond(3, tor10.BD_KET, qnums=[[0, 2, 1, 0],
                                                    [1, 1, -1, 1],
                                                    [2, -1, 1, 0]])
        bd_sym_2 = tor10.Bond(4, tor10.BD_KET, qnums=[[-1, 0, -1, 3],
                                                    [0, 0, -1, 2],
                                                    [1, 0, 1, 0],
                                                    [2, -2, -1, 1]])
        bd_sym_3 = tor10.Bond(2, tor10.BD_BRA, qnums=[[-4, 3, 0, -1],
                                                    [1, 1, -2, 3]])

        sym_T = tor10.UniTensor(bonds=[bd_sym_1, bd_sym_2, bd_sym_3], rowrank=2, labels=[1, 2, 3])

        tqin, tqout = sym_T.GetTotalQnums()

        qin=tor10.Bond(12,tor10.BD_KET, qnums=[[+4, -3, +0, +1], [+3, -1, +2, +0], [+2, -1, +0, +2],
                                               [+1, -1, +0, +3], [+3, -1, -2, +2], [+2, +1, +0, +1],
                                               [+1, +1, -2, +3], [+0, +1, -2, +4], [+2, +0, +0, +1],
                                               [+1, +2, +2, +0], [+0, +2, +0, +2], [-1, +2, +0, +3]]
                           )
        # Output of qnums in Bond.assign is sorted, while it is not sorted in UniTensor.GetTotalQnums
        y=np.lexsort(tqin.qnums.T[::-1])[::-1]
        qnums=tqin.qnums[y,:]

        for i in range(qnums.shape[0]):
            self.assertListEqual(list(qnums[i]),list(qin.qnums[i]))

    def test_GetValidQnums(self):
        bd_sym_1 = tor10.Bond(3, tor10.BD_KET, qnums=[[0], [1], [2]])
        bd_sym_2 = tor10.Bond(4, tor10.BD_KET, qnums=[[-1], [2], [0], [2]])
        bd_sym_3 = tor10.Bond(5, tor10.BD_BRA, qnums=[[4], [2], [2], [5], [1]])
        sym_T = tor10.UniTensor(bonds=[bd_sym_1, bd_sym_2, bd_sym_3], rowrank=2, labels=[10, 11, 12])
        qnums=sym_T.GetValidQnums()
        self.assertListEqual(list(qnums.flatten()),[1, 2, 4])

    def test_PutGetBlock(self):
        bd_sym_1 = tor10.Bond(3, tor10.BD_KET, qnums=[[0], [1], [2]])
        bd_sym_2 = tor10.Bond(4, tor10.BD_KET, qnums=[[-1], [2], [0], [2]])
        bd_sym_3 = tor10.Bond(5, tor10.BD_BRA, qnums=[[4], [2], [2], [5], [1]])

        sym_T= tor10.UniTensor(bonds=[bd_sym_1, bd_sym_2, bd_sym_3], rowrank=2, labels=[10, 11, 12])
        BN=sym_T.GetBlock(2)
        self.assertListEqual(list(BN.Storage.flatten()),[0., 0., 0., 0., 0., 0.])
        self.assertListEqual(list(BN.shape),[3, 2])


        BN.SetElem([float(i) for i in range(6)])
        sym_T.PutBlock(BN,2)

        BN=sym_T.GetBlock(2)
        self.assertListEqual(list(BN.Storage.flatten()),[0., 1., 2., 3., 4., 5.])
        self.assertListEqual(list(BN.shape),[3, 2])

if __name__ == "__main__":
    unittest.main()
