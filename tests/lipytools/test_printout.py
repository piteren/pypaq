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
        pb = ProgBar(total=tot, length=30, fill='X', show_fract=True, show_speed_avg=True, show_eta=True)
        for ix in range(tot):
            time.sleep(random.random()/5)
            pb(ix, prefix='test:', suffix=':ok')

    def test_ProgBar_behaviour(self):

        for ts in [
            0.1,
            1,
            10,
            100,
            1000,
        ]:
            print(f'target speed: {ts}/s')
            tot = int(10 * ts)
            sdt = 1/ts
            pb = ProgBar(total=tot, length=30, guess_speed=ts)
            sd = sdt
            for ix in range(tot):
                sd = sd + sd/20 * (2*random.random()-1)
                sd = min(sdt*3,max(sdt/3, sd))
                time.sleep(sd)
                pb.inc()