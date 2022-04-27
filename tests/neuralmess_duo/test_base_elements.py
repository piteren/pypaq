import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest

import numpy as np
import tensorflow as tf

from pypaq.neuralmess_duo.base_elements import my_initializer, gelu, replace_nan_with_zero


class TestBaseElements(unittest.TestCase):

    def test_my_initializer(self):
        stddev = 0.02
        inp = tf.ones((1,1))
        layer = tf.keras.layers.Dense(10, kernel_initializer=my_initializer(stddev=stddev))
        out = layer(inp)
        std = np.std(out)
        print(f'std of layer is {std}')
        self.assertTrue(stddev-0.01 < std < stddev+0.01)

    def test_gelu(self):
        inp = tf.constant(0.1)
        g = gelu(inp)
        print(f'gelu of 0.1 is {g}')
        self.assertTrue(0.05 < g < 0.06)

    def test_replace_nan_with_zero(self):
        inp = tf.constant(np.nan)
        out = replace_nan_with_zero(inp)
        print(f'input: {inp}, nan replaced: {out}')
        self.assertTrue(out == 0)


if __name__ == '__main__':
    unittest.main()