import numpy as np
import unittest

from pypaq.lipytools.valarr import ValuesArray


class TestValArr(unittest.TestCase):

    def test_base(self):

        va = ValuesArray()
        va.append(1.0)
        va.append(2)
        print(len(va))
        print(va.mean_h95())

        va = ValuesArray(dtype=np.int32)
        va.append(1)
        print(len(va))

        a = va.get_array()
        print(a[0])

    def test_iadd(self):
        va = ValuesArray()
        va += [1.1, 2.2]
        print(len(va), va)
        vb = ValuesArray()
        vb.append(0.1)
        vb += va
        print(len(vb), vb)