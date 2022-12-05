import numpy as np
import random
import unittest

from tests.envy import flush_tmp_dir

from pypaq.comoneural.zeroes_processor import ZeroesProcessor
from pypaq.torchness.base_elements import TBwr

BASE_DIR = f'{flush_tmp_dir()}/comoneural'


class TestZeroesProcessor(unittest.TestCase):

    def setUp(self) -> None:
        flush_tmp_dir()

    def test_ZeroesProcessor(self):

        zepro = ZeroesProcessor(
            intervals=  (10,20,50),
            tbwr=       TBwr(logdir=BASE_DIR))

        for _ in range(2000):

            zsL = []
            z = np.zeros(10, dtype=np.int8)
            if random.random() < 0.1: z[random.randrange(10)] = 1 # random neuron sometimes not activates
            zsL.append(z)
            z = np.zeros(20, dtype=np.int8)
            if random.random() < 0.90: z[1] = 1
            if random.random() < 0.95: z[2] = 1
            if random.random() < 0.99: z[3] = 1
            zsL.append(z)

            zepro.process(zs=zsL)





if __name__ == '__main__':
    unittest.main()