import unittest

from pypaq.lipytools.double_hinge import double_hinge
from pypaq.lipytools.plots import two_dim


class TestDoubleHinge(unittest.TestCase):

    def test_double_hinge(self):

        v = double_hinge(a_value=0.0, b_value=1.0, a_point=0, b_point=10, point=0)
        print(v)
        self.assertTrue(v == 0.0)

        v = double_hinge(a_value=0.0, b_value=1.0, a_point=0, b_point=10, point=10)
        print(v)
        self.assertTrue(v == 1.0)

        v = double_hinge(a_value=0.0, b_value=1.0, a_point=0, b_point=10, point=5)
        print(v)
        self.assertTrue(v == 0.5)

        v = double_hinge(a_value=0.0, b_value=1.0, a_point=0, b_point=10, point=9)
        print(v)
        self.assertTrue(v == 0.9)

        v = double_hinge(a_value=0.0, b_value=1.0, a_point=0, b_point=10, point=-1)
        print(v)
        self.assertTrue(v == 0.0)

        v = double_hinge(a_value=0.0, b_value=1.0, a_point=0, b_point=10, point=20)
        print(v)
        self.assertTrue(v == 1.0)

        v = double_hinge(a_value=1.0, b_value=0.0, a_point=0, b_point=10, point=1)
        print(v)
        self.assertTrue(v == 0.9)

        r = 100
        point = list(range(r))
        vals = [double_hinge(
            a_value=    0.9,
            b_value=   -0.3,
            a_point=    10,
            b_point=    70,
            point=      c) for c in point]
        two_dim(vals)
        vals = [double_hinge(
            a_value=    2,
            b_value=    5,
            a_point=    30,
            b_point=    50,
            point=      c) for c in point]
        two_dim(vals)
