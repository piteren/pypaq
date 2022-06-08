import unittest

from pypaq.pms.subscriptable import Subscriptable, SubGX

_PSDD = {
    'a': [1,    20],
    'b': [0.0,  1.0],
    'c': ('sth','a','b','c','d','rf','ad')}

_DNA = {
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

        sa = SubGX(name='sa', **_DNA)
        print(sa)
        self.assertTrue(sa['a'] == 1)
        self.assertTrue(not sa.get_gxable_point())

        sb_dna = SubGX.gx_dna(sa)
        sb = SubGX(**sb_dna)
        print(sb)
        self.assertTrue(sb['name'] == 'sa_(sgxp)')
        self.assertTrue(sb['a'] == 1)
        self.assertTrue(not sa.get_gxable_point())

        sa['psdd'] = _PSDD
        sc_dna = SubGX.gx_dna(sa,sb)
        sc = SubGX(**sc_dna)
        print(sc)
        self.assertTrue(1<=sc['a']<=20 and 0<=sc['b']<=1 and sc['c'] in _PSDD['c'])
        self.assertTrue(len(sc.get_gxable_point())==3)

        sa['a'] = 1000
        self.assertRaises(AssertionError, SubGX.gx_dna, sa)
        print('\nPoint out of space while GX!')
        sa['a'] = 1

        sa['family'] = 'fa'
        sc['family'] = 'fb'
        self.assertRaises(AssertionError, SubGX.gx_dna, sa, sc)
        print('\nIncompatible families!')

        sc['family'] = 'fa'
        sd_dna = SubGX.gx_dna(sa, sc, name_child='sd', prob_noise=0.9, noise_scale=0.5, prob_diff_axis=0.5)
        sd = SubGX(**sd_dna)
        print(sd)
        self.assertTrue(1<=sd['a']<=20 and 0<=sd['b']<=1 and sd['c'] in _PSDD['c'])
        self.assertTrue(len(sd.get_gxable_point())==3)


if __name__ == '__main__':
    unittest.main()