import random
import time
import unittest

from pypaq.lipytools.printout import stamp, print_nested_dict, ProgBar


class TestLittleMethods(unittest.TestCase):

    def test_stamp(self):
        print(stamp())
        print(stamp(letters=None))
        print(stamp(year=True))
        print(stamp(month=False, day=False))

    def test_print_nested_dict(self):
        dc = {
            'a0': {
                'a1': {
                    'a2': ['el1','el2']
                }
            },
            'b0': ['el1','el2','el3']
        }
        print_nested_dict(dc)

    def test_ProgBar(self):
        tot = 100
        pb = ProgBar(total=tot, length=30, fill='X', show_fract=True, show_speed=True, show_eta=True)
        for ix in range(tot):
            time.sleep(random.random()/5)
            pb(ix, prefix='test:', suffix=':ok')