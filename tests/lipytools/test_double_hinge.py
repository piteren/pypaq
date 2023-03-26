import unittest

from pypaq.lipytools.double_hinge import double_hinge
from pypaq.lipytools.plots import two_dim


class Test_double_hinge(unittest.TestCase):

    def test_double_hinge(self):

        r = 100
        counter = list(range(r))
        vals = [double_hinge(
            s_val=      0.9,
            e_val=      -0.3,
            sf=         0.1,
            ef=         0.3,
            counter=    c,
            max_count=  r) for c in counter]
        two_dim(vals)

        v = double_hinge(0.9,-0.3,0.1,0.3,10,100)
        print(v)
        self.assertTrue(v==0.9)