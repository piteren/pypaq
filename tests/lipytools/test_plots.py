import numpy as np
import unittest

from pypaq.lipytools.plots import histogram, two_dim, three_dim, two_dim_multi


class TestPlots(unittest.TestCase):

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

    def test_two_dim_multi(self):

        va = [1,2,3,1,2,3,1,2,1]
        vb = [1,2,3,4,5,4,3,2,1]
        two_dim_multi(ys=[va,vb])
        va = [22, 22, 14, 10, 12, 4, 6, 14]
        vb = [22, 22.0, 18.0, 14.0, 13.0, 8.5, 7.2, 10.6]
        two_dim_multi(ys=[va, vb])


    # TODO: implement pos encoding with torchness -> enable test
    """
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
    """