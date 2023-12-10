import unittest

from tests.envy import flush_tmp_dir

from pypaq.lipytools.files import prep_folder
from pypaq.pms.parasave import ParaSave, ParaSaveException

PARASAVE_TOPDIR = f'{flush_tmp_dir()}/parasave'
ParaSave.SAVE_TOPDIR = PARASAVE_TOPDIR


_POINT = {'name':'pio', 'a':1, 'b':2, 'c':'nap'}
_PSDD = {'a':[0, 100], 'b':[0.0, 10]}


class TestParaSave(unittest.TestCase):

    def setUp(self) -> None:
        print(f'setting up folder for tests: {PARASAVE_TOPDIR}')
        prep_folder(PARASAVE_TOPDIR, flush_non_empty=True)

    def test_init(self):
        ps = ParaSave(name='ps_test', loglevel=10)
        print(ps)

    def test_save_raises(self):
        ParaSave.SAVE_TOPDIR = None
        ps = ParaSave(name='ps_test')
        print(ps)
        self.assertRaises(ParaSaveException, ps.save_point)
        ParaSave.SAVE_TOPDIR = PARASAVE_TOPDIR

    def test_init_save_load(self):

        ps = ParaSave(name='ps_testA', param='any')
        print(ps)
        # TODO: ps.save() ?
        ps.save_point()
        self.assertTrue(ps.save_topdir == PARASAVE_TOPDIR)

        sd = f'{PARASAVE_TOPDIR}/subdir'
        ps = ParaSave(name='ps_testB', save_topdir=sd)
        print(ps)
        ps.save_point()
        self.assertTrue(ps.save_topdir == sd)

        ps = ParaSave(name='ps_testA')
        print(ps)
        self.assertTrue(ps['param'] == 'any')

    def test_params_resolution(self):

        psa = ParaSave(
            name=       'ps_test',
            family=     'a',
            param_a=    'a',
            param_b=    'b',
            gxable=     False,
            #loglevel=   10,
        )
        print(psa)
        psa.save_point()
        self.assertTrue(psa.family == 'a' and psa.param_a == 'a' and psa.param_b == 'b')

        psb = ParaSave(
            name=           'ps_test',
            assert_saved=   True,
            param_a=        'aa',
            param_c=        'c',
            gxable=         True,
            #loglevel=       10,
        )
        print(psb)

    def test_update(self):

        ps = ParaSave(name='ps_testA', param='any')
        print(ps)
        ps.update({'another':'many'})
        print(ps)
        self.assertTrue(ps.another == 'many')

    def test_update_locked(self):

        ps = ParaSave(name='ps_testA', lock_managed_params=True, param='any', loglevel=10)
        print(ps)
        ps.update({'another':'many'})
        print(ps)
        self.assertTrue(ps.another == 'many')
        self.assertTrue('another' not in ps.get_point())

    def test_more(self):

        # build A and try to save
        ParaSave.SAVE_TOPDIR = None
        ps_point = {}
        ps_point.update(_POINT)
        ps_point['name'] = 'ps'
        ps = ParaSave(**ps_point)
        print(ps.get_point())
        self.assertTrue(ps['a'] == 1)
        #ps.save_point()
        self.assertRaises(ParaSaveException, ps.save_point)
        print('Cannot save without a folder!')
        ParaSave.SAVE_TOPDIR = PARASAVE_TOPDIR

        # build and save A
        ps_point = {}
        ps_point.update(_POINT)
        ps_point['name'] = 'ps'
        ps = ParaSave(**ps_point)
        print(ps.get_point())
        self.assertTrue(ps['a'] == 1, ps['family'] == '__gx-fam__')
        ps['a'] = 2
        ps.save_point()

        # load A from folder
        ps = ParaSave(name='ps')
        print(ps.get_point())
        self.assertTrue(ps['a'] == 2)
        ps['psdd'].update(_PSDD)
        print(ps.get_point())
        ps.save_point()

        # make copy of A to B
        ParaSave.copy_saved_point(
            name_src=   'ps',
            name_trg=   'psb')
        psb = ParaSave(name='psb')
        print(psb.get_point())
        self.assertTrue(psb['a'] == 2)

        # GX saved C from B
        ParaSave.gx_saved_point(
            name_parentA=   'psb',
            name_parentB=   None,
            name_child=     'psc')
        psc = ParaSave(name='psc')
        print(psc.get_point())
        self.assertTrue(0<=psc['a']<=100 and 0<=psc['b']<=10)

        # GX saved D from B & C
        ParaSave.gx_saved_point(
            name_parentA=   'psb',
            name_parentB=   'psc',
            name_child=     'psd')
        psd = ParaSave(name='psd')
        print(psd.get_point())
        self.assertTrue(0<=psd['a']<=100 and 0<=psd['b']<=10)

    def test_parents_GX(self):

        # build and save A
        pa_point = {}
        pa_point.update(_POINT)
        pa_point.update({
            'name':     'pa',
            'family':   'a',
            'psdd':     _PSDD})
        pa = ParaSave(**pa_point)
        pa.save_point()

        # build and save B
        pb_point = {}
        pb_point.update(_POINT)
        pb_point.update({
            'name':     'pb',
            'family':   'a',
            'psdd':     _PSDD})
        pb = ParaSave(**pb_point)
        pb.save_point()

        pc_point = ParaSave.gx_point(parentA=pa, parentB=pb)
        self.assertTrue(pc_point['parents'] == ['pa', 'pb'])
        pc = ParaSave(**pc_point)
        pc.save_point()

        ParaSave.gx_saved_point(
            name_parentA=   'pa',
            name_parentB=   pc.name,
            name_child=     'pd')
        pd = ParaSave(name='pd')
        print(pd)
        self.assertTrue(pd.parents == ['pa', ['pa', 'pb']])