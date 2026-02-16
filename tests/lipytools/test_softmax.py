import numpy as np

from pypaq.lipytools.softmax import softmax


def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    assert abs(result.sum() - 1.0) < 1e-10


def test_softmax_all_equal():
    x = np.array([1.0, 1.0, 1.0])
    result = softmax(x)
    for v in result:
        assert abs(v - 1/3) < 1e-10


def test_softmax_single():
    x = np.array([5.0])
    result = softmax(x)
    assert abs(result[0] - 1.0) < 1e-10


def test_softmax_order():
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    assert result[0] < result[1] < result[2]
