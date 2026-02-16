from pypaq.textools.text_metrics import lev_dist, lev_distL, two_most_distanced


def test_lev_dist():
    assert lev_dist('kitten', 'sitting') == 3
    assert lev_dist('abc', 'abc') == 0
    assert lev_dist('', '') == 0
    assert lev_dist('abc', '') == 3
    assert lev_dist('', 'abc') == 3


def test_lev_distL_strings():
    assert lev_distL('abc', 'abc') == 0
    assert lev_distL('kitten', 'sitting') == 3
    assert lev_distL('', '') == 0


def test_lev_distL_lists():
    assert lev_distL([1, 2, 3], [1, 2, 3]) == 0
    assert lev_distL([1, 2, 3], [1, 2, 4]) == 1
    assert lev_distL([1, 2, 3], [4, 5, 6]) == 3
    assert lev_distL([], []) == 0
    assert lev_distL([1], []) == 1
    assert lev_distL([], [1]) == 1


def test_two_most_distanced():
    # 'abc' and 'xyz' should be the most distanced pair
    result = two_most_distanced('abc', 'abd', 'xyz')
    assert 'abc' in result or 'xyz' in result
    assert len(result) == 2

    # identical strings
    result = two_most_distanced('aaa', 'aaa', 'zzz')
    assert 'zzz' in result
