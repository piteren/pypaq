import pytest

from pypaq.lipytools.moving_average import MovAvg
from pypaq.exception import PyPaqException


def test_base():
    ma = MovAvg()
    with pytest.raises(PyPaqException):
        ma()
    v = None
    for ix in range(5):
        v = ma.upd(10)
        print(v)
    assert v == 10


def test_more():
    ma = MovAvg()
    v = 10
    for ix in range(30):
        print(ma.upd(v))
        v += 10


def test_init_value():
    ma = MovAvg(init_value=10, init_weight=10)
    assert ma() == 10
    ma.upd(100)
    print(ma())
    assert ma() == 19
