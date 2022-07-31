import os
import unittest

import numpy as np

from tests.envy import flush_tmp_dir

from pypaq.neuralmess_duo.nemodelduo import NEModelDUO, fwd_graph

MODEL_DIR = f'{flush_tmp_dir()}/nemodelduo'


class TestNEModelDUO(unittest.TestCase):

    def test_init_base(self):
        flush_tmp_dir()
        model = NEModelDUO(
            name=           'pio',
            name_timestamp= True,
            fwd_func=       fwd_graph,
            save_topdir=    MODEL_DIR,
            do_logfile=     False,
            verb=           2)
        self.assertTrue(model['iLR'] == 0.0005)
        self.assertTrue(len(model['name']) > 3)
        model.save()
        model.exit()
        lsn = os.listdir(MODEL_DIR)[0]
        print(lsn)
        self.assertTrue(lsn == model['name'])

    def test_init_read_only(self):
        tmp_dir = flush_tmp_dir()
        model = NEModelDUO(
            name=           'pio',
            name_timestamp= True,
            fwd_func=       fwd_graph,
            save_topdir=    MODEL_DIR,
            read_only=      True,
            verb=           1)
        self.assertRaises(AssertionError, model.save)
        model.exit()
        self.assertTrue(not os.listdir(tmp_dir))

    # TODO: add tests for different scenarios of init with params (user, folder, defaults, ..)

    def test_save_then_load(self):

        flush_tmp_dir()

        model = NEModelDUO(
            name=           'pio',
            fwd_func=       fwd_graph,
            save_topdir=    MODEL_DIR,
            iLR=            0.001,
            do_logfile=     False,
            verb=           0)
        print(model['iLR'])
        model.update_LR(0.03)
        print(model['iLR'])
        model.save()
        model.exit()

        model = NEModelDUO(
            name=           'pio',
            save_topdir=    MODEL_DIR,
            verb=           2)
        print(model['iLR'])
        self.assertTrue(model['iLR'] == 0.03)
        model.exit()


    def test_train(self):

        flush_tmp_dir()

        model = NEModelDUO(
            name=           'pio',
            fwd_func=       fwd_graph,
            save_topdir=    MODEL_DIR,
            do_logfile=     False,
            verb=           0)
        print(f'\n{model}')

        data = {
            'in_vec':   np.random.random((10,10)),
            'in_true':  np.random.randint(0,1,10)}
        model.train(data)
        print(f'\n{model}')

        model.save()
        model.exit()

        model = NEModelDUO(
            name=           'pio',
            save_topdir=    MODEL_DIR,
            verb=           0)
        print(f'\n{model}')
        model.exit()


    def test_copy_saved(self):

        flush_tmp_dir()

        # save #1
        model = NEModelDUO(
            name=           'pio',
            fwd_func=       fwd_graph,
            save_topdir=    MODEL_DIR,
            do_logfile=     False,
            seed=           121,
            verb=           0)
        print(model['iLR'])
        model.save()
        model.exit()

        NEModelDUO.copy_saved(
            name_src=           'pio',
            name_trg=           'pir',
            save_topdir_src=    MODEL_DIR)
        self.assertTrue('weights.index' in os.listdir(f'{MODEL_DIR}/pir'))

        model = NEModelDUO(name='pir', save_topdir=MODEL_DIR)
        print(model['iLR'])
        model.exit()


    def test_gx_saved(self):

        flush_tmp_dir()

        psdd = {'iLR':  [0.000001,0.1]}

        # save #1
        model = NEModelDUO(
            name=           'pio',
            fwd_func=       fwd_graph,
            save_topdir=    MODEL_DIR,
            iLR=            0.001,
            do_logfile=     False,
            psdd=           psdd,
            seed=           121,
            verb=           0)
        print(model['iLR'])
        model.save()
        model.exit()

        # save #2
        model = NEModelDUO(
            name=           'pip',
            fwd_func=       fwd_graph,
            save_topdir=    MODEL_DIR,
            iLR=            0.01,
            do_logfile=     False,
            psdd=           psdd,
            seed=           122,
            verb=           0)
        print(model['iLR'])
        model.save()
        model.exit()

        # GX 1
        NEModelDUO.gx_saved(
            name_parent_main=           'pio',
            name_parent_scnd=           None,
            name_child=                 'pir',
            save_topdir_parent_main=    MODEL_DIR)
        self.assertTrue('weights.index' in os.listdir(f'{MODEL_DIR}/pir'))

        # GX 2
        NEModelDUO.gx_saved(
            name_parent_main=           'pio',
            name_parent_scnd=           'pip',
            name_child=                 'pis',
            save_topdir_parent_main=    MODEL_DIR)
        self.assertTrue('weights.index' in os.listdir(f'{MODEL_DIR}/pis'))


if __name__ == '__main__':
    unittest.main()