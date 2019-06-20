import unittest
from context import Tor10
import numpy as np
class TestUniTensorObjects(unittest.TestCase):

#    def setUp(self):
#        self.bds_x = [Tor10.Bond(5), Tor10.Bond(5), Tor10.Bond(3)]
#        self.x = Tor10.UniTensor(bonds=self.bds_x, N_inbond=1, labels=[4, 3, 5])

    def test_SetLabel(self):
        bds_x = [Tor10.Bond(5), Tor10.Bond(5), Tor10.Bond(3)]
        x = Tor10.UniTensor(bonds=bds_x, rowrank=1, labels=[4, 3, 5])
        x.SetLabel(-1, 2)
        self.assertEqual(x.labels[2],-1)

    def test_SetLabels(self):
        bds_x = [Tor10.Bond(5), Tor10.Bond(5), Tor10.Bond(3)]
        x = Tor10.UniTensor(bonds=bds_x, rowrank=1, labels=[4, 3, 5])
        x.SetLabels([3,2,1])
        self.assertListEqual(list(x.labels),[3,2,1])

    def test_Reshape(self):
        bds_x = [Tor10.Bond(6), Tor10.Bond(5), Tor10.Bond(3)]
        x = Tor10.UniTensor(bonds=bds_x, rowrank=1, labels=[4, 3, 5])
        y=x.Reshape([2, 3, 5, 3], new_labels=[1, 2, 3, -1], rowrank=2)
        self.assertListEqual(list(y.labels),[1, 2, 3, -1])

    def test_Reshape_(self):
        bds_x = [Tor10.Bond(6), Tor10.Bond(5), Tor10.Bond(3)]
        x = Tor10.UniTensor(bonds=bds_x, rowrank=1, labels=[4, 3, 5])
        x.Reshape_([2, 3, 5, 3], new_labels=[1, 2, 3, -1], rowrank=2)
        self.assertListEqual(list(x.labels), [1, 2, 3, -1])

    def test_CombineBonds(self):
        bds_x = [Tor10.Bond(5), Tor10.Bond(5), Tor10.Bond(3)]
        x = Tor10.UniTensor(bonds=bds_x, rowrank=2, labels=[4, 3, 5])
        x.CombineBonds([5, 3])
        self.assertListEqual(list(x.shape),[5,15])
        self.assertListEqual(list(x.labels), [4,5])

        y = Tor10.UniTensor(bonds=bds_x, rowrank=2, labels=[4, 3, 5])
        y.CombineBonds([3,5])
        self.assertListEqual(list(y.shape), [15, 5])
        self.assertListEqual(list(y.labels), [3, 4])

        z = Tor10.UniTensor(bonds=bds_x * 2, rowrank=3, labels=[4, 3, 5, 6, 7, 8])
        z.CombineBonds([4, 5, 6])
        self.assertListEqual(list(z.shape), [75, 5, 5, 3])
        self.assertListEqual(list(z.labels), [4, 3, 7, 8])
        self.assertEqual(z.rowrank, 1)

        z2 = Tor10.UniTensor(bonds=bds_x * 2, rowrank=3, labels=[4, 3, 5, 6, 7, 8])
        z2.CombineBonds([4,5,6],permute_back=True)
        self.assertListEqual(list(z2.shape), [75, 5, 5, 3])
        self.assertListEqual(list(z2.labels), [4, 3, 7, 8])
        self.assertEqual(z2.rowrank, 2)

    def test_Permute(self):
        bds_x = [Tor10.Bond(6), Tor10.Bond(5), Tor10.Bond(4), Tor10.Bond(3), Tor10.Bond(2)]
        x = Tor10.UniTensor(bonds=bds_x, rowrank=3, labels=[1, 3, 5, 7, 8])
        x.Permute([0,2,1,4,3])
        self.assertListEqual(list(x.labels),[1, 5, 3, 8, 7])
        self.assertEqual(x.rowrank, 3)
        self.assertListEqual(list(x.shape),[6, 4, 5, 2 ,3])
        self.assertFalse(x.is_contiguous())


        y=Tor10.UniTensor(bonds=bds_x, rowrank=3, labels=[1, 3, 5, 7, 8])
        y.Permute([3,1,5,7,8],by_label=True)
        self.assertListEqual(list(y.labels),[3, 1, 5, 7, 8])
        self.assertEqual(y.rowrank, 3)
        self.assertListEqual(list(y.shape), [5, 6, 4, 3, 2])
        self.assertFalse(y.is_contiguous())


        z=Tor10.UniTensor(bonds=bds_x, rowrank=3, labels=[1, 3, 5, 7, 8])
        z.Permute([3, 1, 5, 7, 8], rowrank=2, by_label=True)
        self.assertListEqual(list(z.labels),[3, 1, 5, 7, 8])
        self.assertEqual(z.rowrank,2)
        self.assertListEqual(list(z.shape), [5, 6, 4, 3, 2])
        self.assertFalse(z.is_contiguous())

    def test_View(self):
        pass

    def test_GetTotalQnums(self):
        bd_sym_1 = Tor10.Bond(3, Tor10.BD_KET, qnums=[[0, 2, 1, 0],
                                                    [1, 1, -1, 1],
                                                    [2, -1, 1, 0]])
        bd_sym_2 = Tor10.Bond(4, Tor10.BD_KET, qnums=[[-1, 0, -1, 3],
                                                    [0, 0, -1, 2],
                                                    [1, 0, 1, 0],
                                                    [2, -2, -1, 1]])
        bd_sym_3 = Tor10.Bond(2, Tor10.BD_BRA, qnums=[[-4, 3, 0, -1],
                                                    [1, 1, -2, 3]])

        sym_T = Tor10.UniTensor(bonds=[bd_sym_1, bd_sym_2, bd_sym_3], rowrank=2, labels=[1, 2, 3])

        tqin, tqout = sym_T.GetTotalQnums()

        qin=Tor10.Bond(12,Tor10.BD_KET, qnums=[[+4, -3, +0, +1], [+3, -1, +2, +0], [+2, -1, +0, +2],
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
        pass

    def test_PutBlock(self):
        pass

    def test_GetBlock(self):
        pass

if __name__ == "__main__":
    unittest.main()
