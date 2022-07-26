import os
import time
import unittest

from tests.envy import flush_tmp_dir

from pypaq.neuralmess_duo.tbwr import TBwr


class TestTBwr(unittest.TestCase):

    def test_TBwr(self):

        tmp_dir = flush_tmp_dir()
        self.assertTrue(not os.listdir(tmp_dir))
        tbwr = TBwr(logdir=f'{tmp_dir}/tbwr_test')
        self.assertTrue(not os.listdir(tmp_dir))

        val = 1.0
        for s in range(30):
            tbwr.add(val, 'val', s)
            print(f'put val: {val}')
            val += 0.3
            time.sleep(1)

        tbwr.exit()
