import unittest

from pypaq.lipytools.little_methods import prep_folder
from pypaq.pms.parasave import ParaSave

TEMP_DIR = '_temp_tests/parasave'

_DNA = {
    'name': 'pio',
    'a':    1,
    'b':    2,
    'c':    'nap'}
_PSDD = {
    'a':    [0,100],
    'b':    [0.0,10]}


class TestParaSave(unittest.TestCase):

    def setUp(self) -> None:
        prep_folder(TEMP_DIR, flush_non_empty=True)

    def test_base(self):

        # build A and try to save
        psa_dna = {}
        psa_dna.update(_DNA)
        psa_dna['name'] = 'psa'
        psa = ParaSave(**psa_dna)
        print(psa.get_point())
        self.assertTrue(psa['a'] == 1)
        #psa.save_dna()
        self.assertRaises(AssertionError, psa.save_dna)
        print('Cannot save without a folder!')

        # build and save A
        psa_dna = {}
        psa_dna.update(_DNA)
        psa_dna['name'] = 'psa'
        psa = ParaSave(save_topdir=TEMP_DIR, **psa_dna)
        print(psa.get_point())
        self.assertTrue(psa['a'] == 1)
        psa['a'] = 2
        psa.save_dna()

        # load A from folder
        psa = ParaSave(name='psa', save_topdir=TEMP_DIR)
        print(psa.get_point())
        self.assertTrue(psa['a'] == 2)
        psa['psdd'].update(_PSDD)
        print(psa.get_point())
        psa.save_dna()

        # make copy of A to B
        ParaSave.copy_saved_dna(
            name_src=           'psa',
            name_trg=           'psb',
            save_topdir_src=    TEMP_DIR)
        psb = ParaSave(name='psb', save_topdir=TEMP_DIR)
        print(psb.get_point())
        self.assertTrue(psb['a'] == 2)

        # GX saved C from B
        ParaSave.gx_saved_dna(
            name_parent_main=           'psb',
            name_parent_scnd=           None,
            name_child=                 'psc',
            save_topdir_parent_main=    TEMP_DIR)
        psc = ParaSave(name='psc', save_topdir=TEMP_DIR)
        print(psc.get_point())
        self.assertTrue(0<=psc['a']<=100 and 0<=psc['b']<=10)

        # GX saved D from B & C
        ParaSave.gx_saved_dna(
            name_parent_main=           'psb',
            name_parent_scnd=           'psc',
            name_child=                 'psd',
            save_topdir_parent_main=    TEMP_DIR)
        psd = ParaSave(name='psd', save_topdir=TEMP_DIR)
        print(psd.get_point())
        self.assertTrue(0<=psd['a']<=100 and 0<=psd['b']<=10)


if __name__ == '__main__':
    unittest.main()