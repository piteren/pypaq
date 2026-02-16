import pytest

from pypaq.pms.para import Para, ParaGX
from pypaq.pms.base import PMSException

_PSDD = {
    'a': [1,    20],
    'b': [0.0,  1.0],
    'c': ('sth','a','b','c','d','rf','ad')}

_POINT = {
    'a': 1,
    'b': 0.5,
    'c': 'sth'}


def test_Para_base():

    sa = Para()
    sa['a'] = 1
    sa['_b'] = 2
    assert 'a' in sa
    assert '_b' in sa
    assert 'a' in sa.get_all_params()
    assert '_b' not in sa.get_all_params()
    assert '_b' in sa.get_all_params(protected=True) and len(sa.get_all_params(protected=True)) == 2
    assert 'a' in sa.get_managed_params() and len(sa.get_managed_params())==1
    assert 'a' in sa.get_point() and len(sa.get_point())==1
    assert sa['a']==1

    sa.update({'a':2})
    assert sa['a']==2


def test_Para_check_params_sim():
    init_dict = {
        'alpha':    1,
        'beta':     2,
        'alpha_0':  3}
    sa = Para()
    sa.update(init_dict)

    assert not sa.check_params_sim()

    more_dict = {
        'alpha1':   2,
        'bet':      8}
    assert sa.check_params_sim(params=more_dict)
    assert sa.check_params_sim(params=list(more_dict.keys()))


def test_ParaGX_base():

    with pytest.raises(Exception):
        ParaGX()

    sa = ParaGX(name='sa', **_POINT)
    print(sa)
    assert sa['a'] == 1
    assert not sa.gxable_point


def test_ParaGX_psdd():

    sa = ParaGX(name='sa', psdd=_PSDD, **_POINT)
    print(sa)
    assert sa['a'] == 1
    p = sa.gxable_point
    print(p)
    assert len(p)==3

    sa = ParaGX(psdd=_PSDD)
    print(sa)
    p = sa.gxable_point
    print(p)
    assert len(p)==3


def test_ParaGX_more():

    sa = ParaGX(name='sa', family=None,  **_POINT)
    with pytest.raises(PMSException):
        ParaGX.gx_point(sa)

    sa = ParaGX(name='sa', **_POINT)
    sb_point = ParaGX.gx_point(sa)
    sb = ParaGX(**sb_point)
    print(sb)
    assert sb['name'] == 'sa_(sgxp)'
    assert sb['family'] == '__gx-fam__'
    assert sb['a'] == 1
    assert not sa.gxable_point

    sa['psdd'] = _PSDD
    sc_point = ParaGX.gx_point(sa, sb)
    sc = ParaGX(**sc_point)
    print(sc)
    assert 20 >= sc['a'] >= 1 >= sc['b'] >= 0 and sc['c'] in _PSDD['c']
    assert len(sc.gxable_point)==3

    sa['a'] = 1000
    with pytest.raises(Exception):
        ParaGX.gx_point(sa)
    print('\nPoint out of space while GX!')
    sa['a'] = 1

    sa['family'] = 'fa'
    sc['family'] = 'fb'
    with pytest.raises(Exception):
        ParaGX.gx_point(sa, sc)
    print('\nIncompatible families!')

    sc['family'] = 'fa'
    sd_point = ParaGX.gx_point(
        parentA=        sa,
        parentB=        sc,
        name_child=     'sd',
        prob_mix=       0.3,
        prob_noise=     0.9,
        noise_scale=    0.5,
        prob_diff_axis= 0.5)
    sd = ParaGX(**sd_point)
    print(sd)
    assert 20 >= sd['a'] >= 1 >= sd['b'] >= 0 and sd['c'] in _PSDD['c']
    assert len(sd.gxable_point)==3
