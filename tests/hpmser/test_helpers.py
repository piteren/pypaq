import random
import unittest

from pypaq.hpmser.helpers import str_floatL


class TestHelpers(unittest.TestCase):

    def test_base(self):

        fl = [random.random() for _ in range(5)]
        print(fl)
        print(str_floatL(fl))

        fl = [random.random() for _ in range(10)]
        print(fl)
        print(str_floatL(fl))