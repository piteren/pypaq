import torch
import unittest

from pypaq.torchness.encoders import LayDRT, EncDRT


class TestEncoders(unittest.TestCase):

    def test_LayDRT(self):

        in_width = 100
        inp = torch.rand(in_width) - 0.5
        print(inp)

        lay_drt = LayDRT(
            in_width=   in_width
        )

        lay_drt(inp)