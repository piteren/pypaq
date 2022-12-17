import torch
import unittest

from pypaq.torchness.layers import LayDense, TF_Dropout, LayConv1D, zeroes


class TestLayers(unittest.TestCase):

    def test_lay_dense(self):
        tns = torch.rand(20)
        print(tns)

        dnsl = LayDense(20,10)
        print(dnsl)
        out = dnsl(tns)
        print(out)
        self.assertTrue(out.size()[0] == 10)
        print(torch.sum(out))
        self.assertTrue(float(torch.sum(out)) >= 0.0)

        dnsl = LayDense(
            in_features=    20,
            out_features=   10,
            activation=     None,
            bias=           False,
            initializer=    torch.nn.init.zeros_)
        out = dnsl(tns)
        print(out)
        print(torch.sum(out))
        self.assertTrue(float(torch.sum(out)) == 0.0)

    def test_tf_dropout(self):
        tns = torch.rand((5,5,5))
        sum_tns = float(torch.sum(tns))
        print(sum_tns)
        drl = TF_Dropout(time_drop=0.3, feat_drop=0.3)
        dropped = drl(tns)
        print(dropped)
        sum_dropped = float(torch.sum(dropped))
        print(sum_dropped)

    def test_LayConv1D(self):
        inp = torch.rand(3,6)  # [Channels,SignalSeq]
        print(inp.shape, inp)
        conv_lay = LayConv1D(6,8)
        print(conv_lay)
        out = conv_lay(inp)
        print(out.shape, out)
        self.assertTrue(tuple(out.shape) == (3,8))

    def test_zeroes(self):

        tns = torch.rand(5)
        z = zeroes(tns)
        print(z, z.shape)
        self.assertTrue(z.shape[-1]==5)

        tns = torch.rand((3,4,5))
        z = zeroes(tns)
        print(z, z.shape)
        self.assertTrue(z.shape[-1]==5)

        drl = TF_Dropout(time_drop=0.3, feat_drop=0.3)
        dropped = drl(tns)
        print(dropped)
        print(zeroes(dropped))