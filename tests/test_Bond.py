import unittest
from context import Tor10


class TestBondObjects(unittest.TestCase):


    def setUp(self):

        # Bond:
        # =======================================

        # Non-symmetry:
        self.bd_x = Tor10.Bond(3)
        self.bd_y = Tor10.Bond(4)
        self.bd_z = Tor10.Bond(3)

        # > U1
        self.bd_sym_U1 = Tor10.Bond(3, qnums=[[-1], [0], [1]])

        # > Z2
        self.bd_sym_Z2 = Tor10.Bond(3, qnums=[[0], [1], [0]], sym_types=[Tor10.Symmetry.Zn(2)])

        # > Z4
        self.bd_sym_Z4 = Tor10.Bond(3, qnums=[[0], [2], [3]], sym_types=[Tor10.Symmetry.Zn(4)])

        # > Multiple U1
        self.bd_sym_multU1 = Tor10.Bond(3, qnums=[[-2, -1, 0, -1],
                                             [1, -4, 0, 0],
                                             [-8, -3, 1, 5]])

        # > Multiple mix symmetry: U1 x Z2 x Z4
        self.bd_sym_mix = Tor10.Bond(3, qnums=[[-2, 0, 0],
                                          [-1, 1, 3],
                                          [1, 0, 2]],
                                sym_types=[Tor10.Symmetry.U1(),
                                           Tor10.Symmetry.Zn(2),
                                           Tor10.Symmetry.Zn(4)])

    def test_equality(self):

        self.assertEqual(self.bd_x, self.bd_z)
        self.assertNotEqual(self.bd_x,self.bd_y)

    def test_identity(self):
        self.assertFalse(self.bd_x is self.bd_z)

    def test_assign(self):
        pass


if __name__=="__main__":
    unittest.main()