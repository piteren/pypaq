import time
import unittest

from pypaq.lipytools.decorators import timing, args


class TestDecorators(unittest.TestCase):

    def test_timing(self):

        @timing
        def some_func():
           time.sleep(2)

        some_func()


    def test_args(self):

        @args
        def some_func(
                a,
                b,
                c=5,
                d=False,
        ):
            pass

        some_func(a=1.1, b=3.3, c=8)


    def test_args_more(self):

        @args
        def do_something(
                parameter_long_name_a,
                pa_b,
                pa_c,
                def_a=  'sdgsegegassssssssssssssssssdfgasegsdffffsggggggggggggggggggggggggagsdgsdgsdfdddddddddddd',
                def_b=  31.2434,
                verb=   1,
                **kwargs):
            pass

        d = {'agsgssd':134, 'sdghsdhsdhdhdhg':12354, 'asgagafsdgasdfafasfdasdf':12345155}
        do_something(10, 11, pa_c=d, def_b=5, oth_a=6, first=0.000025)