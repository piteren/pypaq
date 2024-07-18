import unittest

from pypaq.lipytools.double_hinge import double_hinge
from pypaq.lipytools.plots import two_dim


class Test_double_hinge(unittest.TestCase):

    def test_double_hinge(self):

        v = double_hinge(s_val=0.0, e_val=1.0, a=0, b=10, counter=0)
        print(v)
        self.assertTrue(v == 0.0)

        v = double_hinge(s_val=0.0, e_val=1.0, a=0, b=10, counter=10)
        print(v)
        self.assertTrue(v == 1.0)

        v = double_hinge(s_val=0.0, e_val=1.0, a=0, b=10, counter=5)
        print(v)
        self.assertTrue(v == 0.5)

        v = double_hinge(s_val=0.0, e_val=1.0, a=0, b=10, counter=9)
        print(v)
        self.assertTrue(v == 0.9)

        v = double_hinge(s_val=0.0, e_val=1.0, a=0, b=10, counter=-1)
        print(v)
        self.assertTrue(v == 0.0)

        v = double_hinge(s_val=0.0, e_val=1.0, a=0, b=10, counter=20)
        print(v)
        self.assertTrue(v == 1.0)

        v = double_hinge(s_val=1.0, e_val=0.0, a=0, b=10, counter=1)
        print(v)
        self.assertTrue(v == 0.9)

        r = 100
        counter = list(range(r))
        vals = [double_hinge(
            s_val=      0.9,
            e_val=      -0.3,
            a=          10,
            b=          70,
            counter=    c) for c in counter]
        two_dim(vals)
        vals = [double_hinge(
            s_val=      2,
            e_val=      5,
            a=          30,
            b=          50,
            counter=    c) for c in counter]
        two_dim(vals)
