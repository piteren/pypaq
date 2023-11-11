import unittest

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


class TestPara(unittest.TestCase):

    def test_base(self):

        sa = Para()
        sa['a'] = 1
        sa['_b'] = 2
        self.assertTrue('a' in sa)
        self.assertTrue('_b' in sa)
        self.assertTrue('a' in sa.get_all_params())
        self.assertTrue('_b' not in sa.get_all_params())
        self.assertTrue('_b' in sa.get_all_params(protected=True) and len(sa.get_all_params(protected=True)) == 2)
        self.assertTrue('a' in sa.get_managed_params() and len(sa.get_managed_params())==1)
        self.assertTrue('a' in sa.get_point() and len(sa.get_point())==1)
        self.assertTrue(sa['a']==1)

        sa.update({'a':2})
        self.assertTrue(sa['a']==2)

    def test_check_params_sim(self):
        init_dict = {
            'alpha':    1,
            'beta':     2,
            'alpha_0':  3}
        sa = Para()
        sa.update(init_dict)
        #print(sa)

        self.assertTrue(not sa.check_params_sim())

        more_dict = {
            'alpha1':   2,
            'bet':      8}
        self.assertTrue(sa.check_params_sim(params=more_dict))
        self.assertTrue(sa.check_params_sim(params=list(more_dict.keys())))


class TestParaGX(unittest.TestCase):

    def test_base(self):

        self.assertRaises(Exception, ParaGX)

        sa = ParaGX(name='sa', **_POINT)
        print(sa)
        self.assertTrue(sa['a'] == 1)
        self.assertTrue(not sa.gxable_point)


    def test_psdd(self):

        sa = ParaGX(name='sa', psdd=_PSDD, **_POINT)
        print(sa)
        self.assertTrue(sa['a'] == 1)
        p = sa.gxable_point
        print(p)
        self.assertTrue(len(p)==3)

        sa = ParaGX(psdd=_PSDD)
        print(sa)
        p = sa.gxable_point
        print(p)
        self.assertTrue(len(p)==3)


    def test_more(self):

        sa = ParaGX(name='sa', **_POINT)
        self.assertRaises(PMSException, ParaGX.gx_point, sa)

        sa = ParaGX(name='sa', family='c', **_POINT)
        sb_point = ParaGX.gx_point(sa)
        sb = ParaGX(**sb_point)
        print(sb)
        self.assertTrue(sb['name'] == 'sa_(sgxp)')
        self.assertTrue(sb['family'] == 'c')
        self.assertTrue(sb['a'] == 1)
        self.assertTrue(not sa.gxable_point)

        sa['psdd'] = _PSDD
        sc_point = ParaGX.gx_point(sa, sb)
        sc = ParaGX(**sc_point)
        print(sc)
        self.assertTrue(20 >= sc['a'] >= 1 >= sc['b'] >= 0 and sc['c'] in _PSDD['c'])
        self.assertTrue(len(sc.gxable_point)==3)

        sa['a'] = 1000
        self.assertRaises(Exception, ParaGX.gx_point, sa)
        print('\nPoint out of space while GX!')
        sa['a'] = 1

        sa['family'] = 'fa'
        sc['family'] = 'fb'
        self.assertRaises(Exception, ParaGX.gx_point, sa, sc)
        print('\nIncompatible families!')

        sc['family'] = 'fa'
        sd_point = ParaGX.gx_point(
            parentA=    sa,
            parentB=    sc,
            name_child=     'sd',
            prob_mix=       0.3,
            prob_noise=     0.9,
            noise_scale=    0.5,
            prob_diff_axis= 0.5)
        sd = ParaGX(**sd_point)
        print(sd)
        self.assertTrue(20 >= sd['a'] >= 1 >= sd['b'] >= 0 and sd['c'] in _PSDD['c'])
        self.assertTrue(len(sd.gxable_point)==3)