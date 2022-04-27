import numpy as np
import unittest

from pypaq.lipytools.plots import histogram, two_dim, three_dim


class TestStamp(unittest.TestCase):

    def test_histogram(self):
        histogram([1, 4, 5, 5, 3, 3, 5, 65, 32, 45, 5, 5, 6, 33, 5])

    def test_two_dim(self):
        two_dim([0,2,3,4,6,4,7,2,6,3,4])

    def test_two_dim_more(self):

        n = 100

        x = (np.arange(n) - n // 2) / (n / 2 / np.pi / 3)
        y = np.sin(x)

        two_dim(y, x)
        two_dim(list(zip(y, x)))
        two_dim(y)

    def test_three_dim(self):

        from pypaq.neuralmess.layers import positional_encoding

        width = 5

        pe = positional_encoding(90, width, 0.9, 7, verb=0)
        pe = np.squeeze(pe)

        xyz = []
        for rix in range(pe.shape[0]):
            for eix in range(pe.shape[1]):
                xyz.append([rix, eix, pe[rix, eix]])

        three_dim(xyz)




if __name__ == '__main__':
    unittest.main()