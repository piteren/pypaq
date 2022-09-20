import torch
import unittest

from pypaq.torchness.layers import LayDense, TF_Dropout


class TestLayers(unittest.TestCase):

    def test_lay_dense(self):
        tns = torch.rand(100)
        dnsl = LayDense(100, 10)
        out = dnsl(tns)
        print(out)
        self.assertTrue(out.size()[0] == 10)

    def test_tf_dropout(self):
        tns = torch.rand((5,5,5))
        drl = TF_Dropout(time_drop=0.3, feat_drop=0.3)
        dropped = drl(tns)
        print(dropped)
        self.assertFalse(torch.equal(tns, dropped))
