import numpy
import torch
import unittest

from pypaq.torchness.encoders import LayDRT, EncDRT


class TestEncoders(unittest.TestCase):

    def test_LayDRT(self):

        in_width = 10
        inp = torch.rand(in_width) - 0.5
        print(inp, inp.dtype)

        lay_drt = LayDRT(
            in_width=       in_width,
            do_scaled_dns=  True,
            lay_dropout=    0.1,
            res_dropout=    0.1)
        print(lay_drt)

        out = lay_drt(inp)
        print(out)