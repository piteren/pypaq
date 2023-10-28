import unittest

from pypaq.lipytools.moving_average import MovAvg
from pypaq.exception import PyPaqException


class TestMovAvg(unittest.TestCase):

    def test_base(self):

        ma = MovAvg()
        self.assertRaises(PyPaqException, ma)

        v = None
        for ix in range(5):
            v = ma.upd(10)
            print(v)
        self.assertTrue(v == 10)

    def test_more(self):

        ma = MovAvg()
        v = 10
        for ix in range(30):
            print(ma.upd(v))
            v += 10