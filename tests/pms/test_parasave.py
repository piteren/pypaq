import unittest

from tests.envy import flush_tmp_dir

from pypaq.lipytools.files import prep_folder
from pypaq.pms.parasave import ParaSave, ParaSaveException

PARASAVE_DIR = f'{flush_tmp_dir()}/parasave'

DNA = {
    'name': 'pio',
    'a':    1,
    'b':    2,
    'c':    'nap'}
PSDD = {
    'a':    [0,100],
    'b':    [0.0,10]}


class TestParaSave(unittest.TestCase):

    def setUp(self) -> None:
        print(f'setting up folder for tests: {PARASAVE_DIR}')
        prep_folder(PARASAVE_DIR, flush_non_empty=True)


    def test_init(self):
        ps = ParaSave(name='ps_test')
        print(ps)
        self.assertRaises(ParaSaveException, ps.save)


    def test_init_save(self):
        ps = ParaSave(
            name=           'ps_test',
            save_topdir=    PARASAVE_DIR)
        print(ps)
        ps.save()


    def test_saved_params_resolution(self):

        ps = ParaSave(
            name=           'ps_test',
            save_topdir=    PARASAVE_DIR,
            param_a=        'a',
            param_b=        'b',
            loglevel=       10)
        print(ps)
        ps.save()

        psb = ParaSave(
            name=           'ps_test',
            save_topdir=    PARASAVE_DIR,
            assert_saved=   True,
            param_a=        'aa',
            param_c=        'c',
            loglevel=       10)
        print(psb)


    def test_base(self):

        # build A and try to save
        ps_dna = {}
        ps_dna.update(DNA)
        ps_dna['name'] = 'ps'
        ps = ParaSave(**ps_dna)
        print(ps.get_point())
        self.assertTrue(ps['a'] == 1)
        #ps.save_dna()
        self.assertRaises(ParaSaveException, ps.save_dna)
        print('Cannot save without a folder!')

        # build and save A
        ps_dna = {}
        ps_dna.update(DNA)
        ps_dna['name'] = 'ps'
        ps = ParaSave(save_topdir=PARASAVE_DIR, **ps_dna)
        print(ps.get_point())
        self.assertTrue(ps['a'] == 1)
        ps['a'] = 2
        ps.save_dna()

        # load A from folder
        ps = ParaSave(name='ps', save_topdir=PARASAVE_DIR)
        print(ps.get_point())
        self.assertTrue(ps['a'] == 2)
        ps['psdd'].update(PSDD)
        print(ps.get_point())
        ps.save_dna()

        # make copy of A to B
        ParaSave.copy_saved_dna(
            name_src=           'ps',
            name_trg=           'psb',
            save_topdir_src=    PARASAVE_DIR)
        psb = ParaSave(name='psb', save_topdir=PARASAVE_DIR)
        print(psb.get_point())
        self.assertTrue(psb['a'] == 2)

        # GX saved C from B
        ParaSave.gx_saved_dna(
            name_parent_main=           'psb',
            name_parent_scnd=           None,
            name_child=                 'psc',
            save_topdir_parent_main=    PARASAVE_DIR)
        psc = ParaSave(name='psc', save_topdir=PARASAVE_DIR)
        print(psc.get_point())
        self.assertTrue(0<=psc['a']<=100 and 0<=psc['b']<=10)

        # GX saved D from B & C
        ParaSave.gx_saved_dna(
            name_parent_main=           'psb',
            name_parent_scnd=           'psc',
            name_child=                 'psd',
            save_topdir_parent_main=    PARASAVE_DIR)
        psd = ParaSave(name='psd', save_topdir=PARASAVE_DIR)
        print(psd.get_point())
        self.assertTrue(0<=psd['a']<=100 and 0<=psd['b']<=10)