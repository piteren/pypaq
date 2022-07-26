import math
import random
import time
import unittest

from pypaq.lipytools.little_methods import prep_folder
from pypaq.mpython.mptools import DevicesParam
from pypaq.hpmser.search_function import hpmser

TEMP_DIR = '_temp_tests/hpmser'


class TestHpmser(unittest.TestCase):

    def setUp(self) -> None:
        prep_folder(TEMP_DIR, flush_non_empty=True)

    def test_simple_run(self):

        n_proc =            30#3
        av_time =           1#3  # avg num of seconds for calculation
        exception_prob =    0.01#0.3
        verb =              1

        def some_func(
                name :str,
                device :DevicesParam,
                a :int,
                b :float,
                c :float,
                d :float,
                wait=   0.1,
                verb=   0):
            if random.random() < exception_prob: raise Exception('RandomException')
            val = math.sin(b-a*c) - abs(a+3.1)/(d+0.5) - pow(b/2,2)/12
            time.sleep(random.random()*wait)
            if verb>0 :print(f'... {name} calculated on {device} ({a}) ({b}) >> ({val})')
            return val

        func_const = {
            'name':'pio',
            'wait': av_time*2,
            'verb': verb-1}

        psdd = {
            'a':    [-5,    5],
            'b':    [-5.0,  5.0],
            'c':    [-2.0,  2],
            'd':    [0.0,   5]}

        hpmser(
            func=               some_func,
            func_psdd=          psdd,
            func_const=         func_const,
            devices=            [None] * n_proc,
            hpmser_FD=          f'{TEMP_DIR}/_hpmser_runs',
            raise_exceptions=   False,
            verb=               verb)


if __name__ == '__main__':
    unittest.main()