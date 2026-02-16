import numpy as np

from pypaq.lipytools.stats import mam, msmx, stats_pd


def test_mam():
    assert mam([1, 2, 3]) == (1, 2.0, 3)
    assert mam([5]) == (5, 5.0, 5)
    assert mam([]) == (0.0, 0.0, 0.0)
    assert mam([-3, 0, 3]) == (-3, 0.0, 3)


def test_msmx():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = msmx(vals)
    assert r['min'] == 1.0
    assert r['max'] == 5.0
    assert r['mean'] == 3.0
    assert r['median'] == 3.0
    assert r['std'] > 0
    assert r['sem'] > 0
    assert r['h95'] > 0
    assert r['L2norm'] > 0
    assert 'string' in r


def test_msmx_numpy():
    arr = np.array([10.0, 20.0, 30.0])
    r = msmx(arr)
    assert r['mean'] == 20.0
    assert r['min'] == 10.0
    assert r['max'] == 30.0


def test_msmx_no_scin():
    r = msmx([1.0, 2.0, 3.0], use_scin=False)
    assert 'mean' in r['string']


def test_stats_pd():
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    s = stats_pd(vals)
    assert 'mean' in s
    assert 'std' in s

    s5 = stats_pd(vals, n_percentiles=5)
    print(s5)
