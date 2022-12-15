import torch
import unittest

from pypaq.torchness.tbwr import TBwr

from tests.envy import flush_tmp_dir

TBWR_DIR = f'{flush_tmp_dir()}/tbwr'


class TestTBwr(unittest.TestCase):

    def setUp(self) -> None:
        flush_tmp_dir()

    def test_TBwr_values(self):

        tbwr = TBwr(logdir=f'{TBWR_DIR}/val')

        val = 1.7
        for ix in range(100):
            tbwr.add(value=val, tag='val', step=ix)
            val += 0.15

    def test_TBwr_histogram(self):

        tbwr = TBwr(logdir=f'{TBWR_DIR}/values_histogram')

        vals = torch.rand(100)
        for ix in range(100):
            tbwr.add_histogram(values=vals, tag='vals_hist', step=ix)
            vals = vals/1.05 + torch.rand(100)*0.05

