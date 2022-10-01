import unittest

from tests.envy import flush_tmp_dir

from pypaq.mpython.mpdecor import proc_wait
from pypaq.neuralmess.nemodel import NEModel, fwd_graph

NEMODEL_DIR = f'{flush_tmp_dir()}/nemodel'

DNA = {'seq_len':20, 'emb_num':33, 'seed':111}


class TestNEModel(unittest.TestCase):

    def test_init(self):

        flush_tmp_dir()

        nnm = NEModel(
            name=           'nemodel_test_A',
            fwd_func=       fwd_graph,
            opt_func=       None,
            save_topdir=    NEMODEL_DIR,
            do_logfile=     False,  # INFO: unittests crashes with logger
            verb=           1,
            **DNA)
        nnm.save_ckpt()
        self.assertTrue(nnm['opt_func'] is None and nnm['iLR'] == 0.003)

        nnm = NEModel(
            name=           'nemodel_test_B',
            fwd_func=       fwd_graph,
            save_topdir=    NEMODEL_DIR,
            do_logfile=     False,  # INFO: unittests crashes with logger
            verb=           1,
            **DNA)
        self.assertTrue(nnm['opt_func'] is not None and nnm['iLR']==0.003 and 'loss' in nnm)
        nnm.save_ckpt()

        nnm = NEModel(
            name=           'nemodel_test_C',
            fwd_func=       fwd_graph,
            save_topdir=    NEMODEL_DIR,
            do_logfile=     False,      # INFO: unittests crashes with logger
            verb=           1,
            **DNA)
        self.assertTrue(nnm['seed']==111 and nnm['iLR']==0.003 and 'loss' in nnm)
        self.assertTrue('loss' not in nnm.get_managed_params())
        nnm.save()

        print('\nsaved, now loading...')

        nnm = NEModel(
            name=           'nemodel_test_C',
            save_topdir=    NEMODEL_DIR,
            do_logfile=     False,      # INFO: unittests crashes with logger
            verb=           0)
        self.assertTrue(nnm['seq_len']==20 and nnm['emb_num']==33)


    def test_train(self):

        flush_tmp_dir()
        raise NotImplementedError


    def test_GX(self):

        flush_tmp_dir()

        @proc_wait
        def saveAB():
            nnm = NEModel(
                name=           'nemodel_test_A',
                fwd_func=       fwd_graph,
                save_topdir=    NEMODEL_DIR,
                do_logfile=     False,      # INFO: unittests crashes with logger
                verb=           1,
                **DNA)
            nnm.save()
            nnm = NEModel(
                name=           'nemodel_test_B',
                fwd_func=       fwd_graph,
                save_topdir=    NEMODEL_DIR,
                do_logfile=     False,      # INFO: unittests crashes with logger
                verb=           1,
                **DNA)
            nnm.save()

        # INFO: needs to run saveAB() in a subprocess cause TF elements/graphs do conflict
        saveAB()
        NEModel.gx_saved(
            name_parent_main=           'nemodel_test_A',
            name_parent_scnd=           'nemodel_test_B',
            name_child=                 'nemodel_test_C',
            save_topdir_parent_main=    NEMODEL_DIR,
            #do_gx_ckpt=                 False
        )


if __name__ == '__main__':
    unittest.main()