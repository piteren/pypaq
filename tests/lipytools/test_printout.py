import random
import time
import unittest

from pypaq.lipytools.printout import (
    nice_scin, nice_float_pad, nice_float_width,
    stamp, print_nested_dict, ProgBar)


class TestLittleMethods(unittest.TestCase):

    def test_nice_scin(self):
        for n in [
            0.1,
            0.13124,
            0.45738683456,
            0.0021342,
            0.000001234,
            0.0,
            1.3453,
            1.9999999,
            123.12345,
            1,
            2,
            1e2,
            -0.0,
            -1,
            -2,
            -999,
            9999,
            -99999999,
            99999999,
            1.234e14,
            1.234e-14,
            0.0000000013441,
            -0.0000000013441,
        ]:
            print(f"{str(n):20} -> {nice_scin(n):10} {nice_scin(n, precision=2):10} {nice_scin(n, replace_zero=False):10} {nice_scin(n, add_plus=True):10}")

    def test_nice_float_pad(self):
        for n in [
            0.1,
            0.13124,
            0.45738683456,
            0.0021342,
            0.000001234,
            0.0,
            1.3453,
            1.9999999,
            123.12345,
            1,
            2,
            1e2,
            -0.0,
            -1,
            -2,
            -999,
            9999,
            -888888888,
            888888888,
            1.234e14,
            1.234e-14,
            1.234e-140,
            0.0000000013441,
            -0.0000000013441,
        ]:
            s = nice_float_pad(n)
            print(f"{str(n):20} -> {s} ({len(s)})")

        self.assertRaises(ValueError, nice_float_pad, 1, width=4)

    def test_nice_float_width(self):
        for n in [
                # typical values
            0.1,
            0.13124,
            0.4573868,
            0.0021342,
            0.000001234,
            0.0,
            1.3453,
            1.9999999,
                # supported but not designed for
            123.123,
            1,
            2,
            1e2,
            -0.0,
            -1,
            -2,
                # edges
            -999,
            9999,
        ]:
            print(f"{str(n):20} -> {nice_float_width(n)}")

        for n in [
            -1000,
            10000,
            2353252.6,
        ]:
            self.assertRaises(ValueError, nice_float_width, n)

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