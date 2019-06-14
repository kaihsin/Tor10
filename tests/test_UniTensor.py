import unittest
from context import Tor10

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
        pass


    def test_View(self):
        pass

    def test_GetTotalQnums(self):
        pass

    def test_GetValidQnums(self):
        pass

    def test_PutBlock(self):
        pass

    def test_GetBlock(self):
        pass

if __name__ == "__main__":
    unittest.main()
