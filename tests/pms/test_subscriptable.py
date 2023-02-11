import unittest

from pypaq.pms.subscriptable import Subscriptable, SubGX

_PSDD = {
    'a': [1,    20],
    'b': [0.0,  1.0],
    'c': ('sth','a','b','c','d','rf','ad')}

_POINT = {
    'a': 1,
    'b': 0.5,
    'c': 'sth'}


class TestSubscriptable(unittest.TestCase):

    def test_base(self):

        sa = Subscriptable()
        sa['a'] = 1
        sa['_b'] = 2
        self.assertTrue('a' in sa)
        self.assertTrue('_b' in sa)
        self.assertTrue('a' in sa.get_all_fields())
        self.assertTrue('_b' not in sa.get_all_fields())
        self.assertTrue('_b' in sa.get_all_fields(protected=True) and len(sa.get_all_fields(protected=True))==2)
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
        sa = Subscriptable()
        sa.update(init_dict)
        #print(sa)

        self.assertTrue(not sa.check_params_sim())

        more_dict = {
            'alpha1':   2,
            'bet':      8}
        self.assertTrue(sa.check_params_sim(params=more_dict))
        self.assertTrue(sa.check_params_sim(params=list(more_dict.keys())))



class TestSubGX(unittest.TestCase):

    def test_base(self):

        sa = SubGX(name='sa', **_POINT)
        print(sa)
        self.assertTrue(sa['a'] == 1)
        self.assertTrue(not sa.get_gxable_point())

        sb_point = SubGX.gx_point(sa)
        sb = SubGX(**sb_point)
        print(sb)
        self.assertTrue(sb['name'] == 'sa_(sgxp)')
        self.assertTrue(sb['a'] == 1)
        self.assertTrue(not sa.get_gxable_point())

        sa['psdd'] = _PSDD
        sc_point = SubGX.gx_point(sa, sb)
        sc = SubGX(**sc_point)
        print(sc)
        self.assertTrue(1<=sc['a']<=20 and 0<=sc['b']<=1 and sc['c'] in _PSDD['c'])
        self.assertTrue(len(sc.get_gxable_point())==3)

        sa['a'] = 1000
        self.assertRaises(AssertionError, SubGX.gx_point, sa)
        print('\nPoint out of space while GX!')
        sa['a'] = 1

        sa['family'] = 'fa'
        sc['family'] = 'fb'
        self.assertRaises(AssertionError, SubGX.gx_point, sa, sc)
        print('\nIncompatible families!')

        sc['family'] = 'fa'
        sd_point = SubGX.gx_point(
            parent_main=    sa,
            parent_scnd=    sc,
            name_child=     'sd',
            prob_mix=       0.3,
            prob_noise=     0.9,
            noise_scale=    0.5,
            prob_diff_axis= 0.5)
        sd = SubGX(**sd_point)
        print(sd)
        self.assertTrue(1<=sd['a']<=20 and 0<=sd['b']<=1 and sd['c'] in _PSDD['c'])
        self.assertTrue(len(sd.get_gxable_point())==3)