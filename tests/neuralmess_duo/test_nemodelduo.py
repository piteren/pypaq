import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest

from tests.envy import flush_tmp_dir

from pypaq.neuralmess_duo.nemodelduo import NEModelDUO, fwd_graph

SCRIPT_TEMP_DIR = f'{flush_tmp_dir()}/nemodelduo'


class TestNEModelDUO(unittest.TestCase):

    def test_init_base(self):
        model = NEModelDUO(
            name=           'pio',
            name_timestamp= True,
            fwd_func=       fwd_graph,
            warm_up=        100,
            verb=           1)
        self.assertTrue(model['iLR'] == 0.0005)
        self.assertTrue(len(model['name']) > 3)

    def test_save_load(self):

        model = NEModelDUO(
            name=           'pio',
            fwd_func=       fwd_graph,
            save_topdir=    SCRIPT_TEMP_DIR,
            iLR=            0.001,
            verb=           0)
        print(model['iLR'])
        model.save()

        model = NEModelDUO(
            name=           'pio',
            fwd_func=       fwd_graph,
            save_topdir=    SCRIPT_TEMP_DIR,
            verb=           1)
        print(model['iLR'])
        self.assertTrue(model['iLR'] == 0.001)


if __name__ == '__main__':
    unittest.main()