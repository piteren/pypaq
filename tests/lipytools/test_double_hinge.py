import pytest

from pypaq.lipytools.double_hinge import double_hinge
from pypaq.lipytools.plots import two_dim


def test_double_hinge():

    v = double_hinge(a_value=0.0, b_value=1.0, a_point=0, b_point=10, point=0)
    print(v)
    assert round(v,2) == 0.0

    v = double_hinge(a_value=0.0, b_value=1.0, a_point=0, b_point=10, point=10)
    print(v)
    assert round(v,2) == 1.0

    v = double_hinge(a_value=0.0, b_value=1.0, a_point=0, b_point=10, point=5)
    print(v)
    assert round(v,2) == 0.5

    v = double_hinge(a_value=0.0, b_value=1.0, a_point=0, b_point=10, point=9)
    print(v)
    assert round(v,2) == 0.9

    v = double_hinge(a_value=0.0, b_value=1.0, a_point=0, b_point=10, point=-1)
    print(v)
    assert round(v,2) == 0.0

    v = double_hinge(a_value=0.0, b_value=1.0, a_point=0, b_point=10, point=20)
    print(v)
    assert round(v,2) == 1.0

    v = double_hinge(a_value=1.0, b_value=0.0, a_point=0, b_point=10, point=1)
    print(v)
    assert round(v,2) == 0.9

    v = double_hinge(a_value=0.9, b_value=-0.3, a_point=10, b_point=70, point=0)
    print(v)
    assert round(v,2) == 0.9
    v = double_hinge(a_value=0.9, b_value=-0.3, a_point=10, b_point=70, point=10)
    print(v)
    assert round(v,2) == 0.9
    v = double_hinge(a_value=0.9, b_value=-0.3, a_point=10, b_point=70, point=20)
    print(v)
    assert round(v,2) == 0.7
    v = double_hinge(a_value=0.9, b_value=-0.3, a_point=10, b_point=70, point=70)
    print(v)
    assert round(v,2) ==  -0.3
    v = double_hinge(a_value=0.9, b_value=-0.3, a_point=10, b_point=70, point=70)
    print(v)
    assert round(v,2) ==  -0.3


    v = double_hinge(a_value=2, b_value=5, a_point=30, b_point=50, point=0)
    print(v)
    assert round(v,2) == 2.0
    v = double_hinge(a_value=2, b_value=5, a_point=30, b_point=50, point=30)
    print(v)
    assert round(v,2) == 2.0
    v = double_hinge(a_value=2, b_value=5, a_point=30, b_point=50, point=45)
    print(v)
    assert round(v,2) == 4.25
    v = double_hinge(a_value=2, b_value=5, a_point=30, b_point=50, point=50)
    print(v)
    assert round(v,2) == 5.0
    v = double_hinge(a_value=2, b_value=5, a_point=30, b_point=50, point=100)
    print(v)
    assert round(v,2) == 5.0

    r = 100
    rng = list(range(r))
    print(rng)
    vals = [
        double_hinge(
            a_value=    0.9,
            b_value=   -0.3,
            a_point=    10,
            b_point=    70,
            point=      p,
        ) for p in rng]
    two_dim(vals)
    print(vals)
    vals = [
        double_hinge(
            a_value=    2,
            b_value=    5,
            a_point=    30,
            b_point=    50,
            point=      p,
        ) for p in rng]
    two_dim(vals)
    print(vals)


def test_assert():
    with pytest.raises(Exception):
        double_hinge(2,5,50,30,0)
