from pypaq.lipytools.opt_process import chunked


def test_chunked_basic():
    result = list(chunked([1, 2, 3, 4, 5], 2))
    assert result == [[1, 2], [3, 4], [5]]


def test_chunked_exact():
    result = list(chunked([1, 2, 3, 4], 2))
    assert result == [[1, 2], [3, 4]]


def test_chunked_single():
    result = list(chunked([1, 2, 3], 1))
    assert result == [[1], [2], [3]]


def test_chunked_larger_than_input():
    result = list(chunked([1, 2], 10))
    assert result == [[1, 2]]


def test_chunked_empty():
    result = list(chunked([], 5))
    assert result == []


def test_chunked_string():
    result = list(chunked('abcdef', 3))
    assert result == ['abc', 'def']
