import unittest

from pypaq.lipytools.moving_average import MovAvg


class TestMovAvg(unittest.TestCase):

    def test_base(self):

        ma = MovAvg()
        self.assertRaises(Exception, ma)

        v = None
        for ix in range(5):
            v = ma.upd(10)
            print(v)
        self.assertTrue(v == 10)

        for ix in range(30):
            v = ma.upd(20)
            d = (20-v)/10
            print(f'{ix:2} {d:.5f} {v:.3f}')
        self.assertTrue(19.55 < v < 19.6)