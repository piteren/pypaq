import unittest

import tensorflow as tf


class TestTF(unittest.TestCase):

    def test_TF(self):
        print(f'\nusing TF: {tf.__version__}, executing_eagerly: {tf.executing_eagerly()}')
        self.assertTrue(tf.executing_eagerly())


if __name__ == '__main__':
    unittest.main()