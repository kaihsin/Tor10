import unittest
from context import Tor10

class TestUniTensorObjects(unittest.TestCase):

    def setUp(self):
        self.bds_x = [Tor10.Bond(5), Tor10.Bond(5), Tor10.Bond(3)]
        self.x = Tor10.UniTensor(bonds=self.bds_x, N_inbond=1, labels=[4, 3, 5])

    def test_SetLabel(self):
        self.x.SetLabel(-1, 2)
        self.assertEqual(self.x.labels[2],-1)

    def test_SetLabels(self):
        self.x.SetLabels([3,2,1])
        self.assertListEqual(list(self.x.labels),[3,2,1])

if __name__ == "__main__":
    unittest.main()
