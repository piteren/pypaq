import unittest

from tests.envy import flush_tmp_dir

from pypaq.lipytools.files import prep_folder
from pypaq.pms.parasave import ParaSave, ParaSaveException

PARASAVE_TOPDIR = f'{flush_tmp_dir()}/parasave'
ParaSave.SAVE_TOPDIR = PARASAVE_TOPDIR


POINT = {
    'name': 'pio',
    'a':    1,
    'b':    2,
    'c':    'nap'}
PSDD = {
    'a':    [0,100],
    'b':    [0.0,10]}


class TestParaSave(unittest.TestCase):

    def setUp(self) -> None:
        print(f'setting up folder for tests: {PARASAVE_TOPDIR}')
        prep_folder(PARASAVE_TOPDIR, flush_non_empty=True)


    def test_init(self):
        ParaSave.SAVE_TOPDIR = None
        ps = ParaSave(name='ps_test')
        print(ps)
        self.assertRaises(ParaSaveException, ps.save_point)
        ParaSave.SAVE_TOPDIR = PARASAVE_TOPDIR


    def test_init_save_load(self):

        ps = ParaSave(name='ps_testA', param='any')
        print(ps)
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

    def test_saved_params_resolution(self):

        ps = ParaSave(
            name=           'ps_test',
            param_a=        'a',
            param_b=        'b',
            loglevel=       10)
        print(ps)
        ps.save_point()

        psb = ParaSave(
            name=           'ps_test',
            assert_saved=   True,
            param_a=        'aa',
            param_c=        'c',
            loglevel=       10)
        print(psb)


    def test_more(self):

        # build A and try to save
        ParaSave.SAVE_TOPDIR = None
        ps_point = {}
        ps_point.update(POINT)
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
        ps_point.update(POINT)
        ps_point['name'] = 'ps'
        ps = ParaSave(**ps_point)
        print(ps.get_point())
        self.assertTrue(ps['a'] == 1)
        ps['a'] = 2
        ps.save_point()

        # load A from folder
        ps = ParaSave(name='ps')
        print(ps.get_point())
        self.assertTrue(ps['a'] == 2)
        ps['psdd'].update(PSDD)
        print(ps.get_point())
        ps.save_point()

        # make copy of A to B
        ParaSave.copy_saved_point(
            name_src=           'ps',
            name_trg=           'psb')
        psb = ParaSave(name='psb')
        print(psb.get_point())
        self.assertTrue(psb['a'] == 2)

        # GX saved C from B
        ParaSave.gx_saved_point(
            name_parent_main=           'psb',
            name_parent_scnd=           None,
            name_child=                 'psc')
        psc = ParaSave(name='psc')
        print(psc.get_point())
        self.assertTrue(0<=psc['a']<=100 and 0<=psc['b']<=10)

        # GX saved D from B & C
        ParaSave.gx_saved_point(
            name_parent_main=           'psb',
            name_parent_scnd=           'psc',
            name_child=                 'psd')
        psd = ParaSave(name='psd')
        print(psd.get_point())
        self.assertTrue(0<=psd['a']<=100 and 0<=psd['b']<=10)