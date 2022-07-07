import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest

import numpy as np
import tensorflow as tf

from tests.envy import flush_tmp_dir

from pypaq.neuralmess_duo.base_elements import my_initializer, gelu, replace_nan_with_zero, TBwr


class TestBaseElements(unittest.TestCase):

    def test_my_initializer(self):
        stddev = 0.02
        inp = tf.ones((1,1))
        layer = tf.keras.layers.Dense(10, kernel_initializer=my_initializer(stddev=stddev))
        out = layer(inp)
        std = np.std(out)
        print(f'\nstd of layer is {std} (taget: {stddev})')
        self.assertTrue(stddev-0.01 < std < stddev+0.01)

    def test_gelu(self):
        inp = tf.constant(0.1)
        g = gelu(inp)
        print(f'\ngelu of 0.1 is {g} (should be: 0.05398)')
        self.assertTrue(0.05 < g < 0.06)

    def test_replace_nan_with_zero(self):
        inp = tf.constant(np.nan)
        out = replace_nan_with_zero(inp)
        print(f'\ninput: {inp}, nan replaced: {out}')
        self.assertTrue(out == 0)

    def test_lr_scaler(self):
        # TODO: write test
        pass

    def test_grad_clipper_AVT(self):
        # TODO: write test
        pass

    def test_TBwr(self):
        tmp_dir = flush_tmp_dir()
        self.assertTrue(not os.listdir(tmp_dir))
        tbwr = TBwr(logdir=f'{tmp_dir}/tbwr')
        self.assertTrue(not os.listdir(tmp_dir))
        val=1.0
        for s in range(10):
            tbwr.add(val,'val',s)
            val += 0.3
        self.assertTrue(os.listdir(tmp_dir)[0]=='tbwr')


if __name__ == '__main__':
    unittest.main()