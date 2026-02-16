import time

from pypaq.lipytools.decorators import timing, args, autoinit


def test_timing():

    @timing
    def some_func():
       time.sleep(2)

    some_func()


def test_args():

    @args
    def some_func(
            a,
            b,
            c=5,
            d=False,
    ):
        pass

    some_func(a=1.1, b=3.3, c=8)


def test_args_more():

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


def test_autoinit():

    class MyClass:
        @autoinit
        def __init__(self, a, b, c=10, d=False):
            pass

    obj = MyClass(1, 2)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 10
    assert obj.d is False


def test_autoinit_kwargs_override():

    class MyClass:
        @autoinit
        def __init__(self, a, b, c=10):
            pass

    obj = MyClass(1, 2, c=99)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 99


def test_autoinit_positional_defaults():

    class MyClass:
        @autoinit
        def __init__(self, a, b, c=10, d=20):
            pass

    obj = MyClass(1, 2, 99)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 99
    assert obj.d == 20
