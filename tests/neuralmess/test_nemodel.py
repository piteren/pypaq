import unittest

from pypaq.lipytools.little_methods import prep_folder
from pypaq.mpython.mpdecor import proc_wait
from pypaq.neuralmess.nemodel import NEModelBase, NEModel

TEMP_DIR = '_temp_tests/nemodel'


class TestNEModel(unittest.TestCase):

    def setUp(self) -> None:
        prep_folder(TEMP_DIR, flush_non_empty=True)
        pass

    def test_init(self):

        dna = {'seq_len':20, 'emb_num':33, 'seed':111}

        nnm = NEModelBase(
            name=           'nemodel_base_test',
            opt_func=       None,
            save_topdir=    TEMP_DIR,
            do_logfile=     False,      # INFO: unittests crashes with logger
            verb=           1,
            **dna)
        nnm.save_ckpt()
        self.assertTrue(nnm['opt_func'] is None and nnm['iLR'] == 0.003)

        nnm = NEModelBase(
            name=           'nemodel_base_test',
            save_topdir=    TEMP_DIR,
            do_logfile=     False,      # INFO: unittests crashes with logger
            verb=           1,
            **dna)
        self.assertTrue(nnm['opt_func'] is not None and nnm['iLR']==0.003 and 'loss' in nnm)
        nnm.save_ckpt()

        nnm = NEModel(
            name=           'nemodel_test',
            save_topdir=    TEMP_DIR,
            do_logfile=     False,      # INFO: unittests crashes with logger
            verb=           1,
            **dna)
        self.assertTrue(nnm['seed']==111 and nnm['iLR']==0.003 and 'loss' in nnm)
        self.assertTrue('loss' not in nnm.get_managed_params())
        nnm.save()

        print('\n@@@ saved, now loading...')

        nnm = NEModel(
            name=           'nemodel_test',
            save_topdir=    TEMP_DIR,
            do_logfile=     False,      # INFO: unittests crashes with logger
            verb=           0)
        self.assertTrue(nnm['seq_len']==20 and nnm['emb_num']==33)

    def test_GX(self):
        @proc_wait
        def saveAB():
            dna = {'seq_len':20, 'emb_num':33, 'seed':111}
            nnm = NEModel(
                name=           'nemodel_testA',
                save_topdir=    TEMP_DIR,
                do_logfile=     False,      # INFO: unittests crashes with logger
                verb=           1,
                **dna)
            nnm.save()
            dna = {'seq_len':20, 'emb_num':33, 'seed':111}
            nnm = NEModel(
                name=           'nemodel_testB',
                save_topdir=    TEMP_DIR,
                do_logfile=     False,      # INFO: unittests crashes with logger
                verb=           1,
                **dna)
            nnm.save()
        # INFO: needs to run saveAB() in a subprocess cause TF elements/graphs do conflict
        saveAB()
        NEModel.gx_saved_dna_cc(
            name_parent_main=           'nemodel_testA',
            name_parent_scnd=           'nemodel_testB',
            name_child=                 'nemodel_testC',
            save_topdir_parent_main=    TEMP_DIR,
            #do_gx_ckpt=                 False
        )


if __name__ == '__main__':
    unittest.main()