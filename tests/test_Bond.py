import unittest
from context import tor10


class TestBondObjects(unittest.TestCase):

    def setUp(self):
        # Bond:
        # =======================================

        # Non-symmetry:
        self.bd_x = tor10.Bond(3)
        self.bd_y = tor10.Bond(4)
        self.bd_z = tor10.Bond(3)

        # > U1
        self.bd_sym_U1 = tor10.Bond(3, qnums=[[-1], [0], [1]])

        # > Z2
        self.bd_sym_Z2 = tor10.Bond(3, qnums=[[0], [1], [0]], sym_types=[tor10.Symmetry.Zn(2)])

        # > Z4
        self.bd_sym_Z4 = tor10.Bond(3, qnums=[[0], [2], [3]], sym_types=[tor10.Symmetry.Zn(4)])

        # > Multiple U1
        self.bd_sym_multU1 = tor10.Bond(3, qnums=[[-2, -1, 0, -1],
                                                  [1, -4, 0, 0],
                                                  [-8, -3, 1, 5]])

        # > Multiple mix symmetry: U1 x Z2 x Z4
        self.bd_sym_mix = tor10.Bond(3, qnums=[[-2, 0, 0],
                                               [-1, 1, 3],
                                               [1, 0, 2]],
                                     sym_types=[tor10.Symmetry.U1(),
                                                tor10.Symmetry.Zn(2),
                                                tor10.Symmetry.Zn(4)])

    def test_equal(self):
        self.assertEqual(self.bd_x, self.bd_z)
        self.assertNotEqual(self.bd_x, self.bd_y)

    def test_identity(self):
        self.assertFalse(self.bd_x is self.bd_z)

    def test_assign(self):
        bond1=tor10.Bond(3)
        bond1.assign(5)
        self.assertEqual(bond1.dim, 5)
        bond1.assign(3,qnums=[[1], [2], [3]])
        self.assertListEqual(list(bond1.qnums), [[1], [2], [3]])


        bond1.assign(3, qnums=[[1], [2], [3]], sym_types=[tor10.Symmetry.Zn(4)])
        self.assertEqual(bond1.sym_types,[tor10.Symmetry.Zn(4)] )

    def test_change(self):
        bond1=tor10.Bond(3)
        bond1.change(tor10.BD_REGULAR)
        self.assertEqual(bond1.bondType,tor10.BD_REGULAR)

    def test_combine(self):

        a = tor10.Bond(3)
        b = tor10.Bond(4)
        c = tor10.Bond(2, qnums=[[0, 1, -1], [1, 1, 0]])
        d = tor10.Bond(2, qnums=[[1, 0, -1], [1, 0, 0]])
        e = tor10.Bond(2, qnums=[[1, 0], [1, 0]], sym_types=[tor10.Symmetry.U1(),tor10.Symmetry.Zn(4)])

        a.combine(b)
        self.assertEqual(a.dim, 12)

        c.combine(d)
        self.assertListEqual(list(d.qnums[0]),[ 1, 0, -1 ])

if __name__ == "__main__":

    unittest.main()
