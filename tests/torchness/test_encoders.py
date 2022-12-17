import torch
import unittest

from pypaq.torchness.encoders import LayDRT, EncDRT, EncCNN


class TestEncoders(unittest.TestCase):

    def test_LayDRT(self):

        in_width = 10
        inp = torch.rand(in_width) - 0.5
        print(inp, inp.dtype)

        lay_drt = LayDRT(in_width)
        print(lay_drt)
        out = lay_drt(inp)
        print(out)
        self.assertTrue(out['out'].shape[-1] == in_width)
        self.assertTrue(out['zsL'][0].shape[-1] == in_width)

        lay_drt = LayDRT(
            in_width=       in_width,
            do_scaled_dns=  True,
            dns_scale=      4,
            lay_dropout=    0.1,
            res_dropout=    0.1)
        print(lay_drt)
        out = lay_drt(inp)
        print(out)
        self.assertTrue(out['out'].shape[-1] == in_width)
        self.assertTrue(out['zsL'][0].shape[-1] == in_width * 4)

        dev = torch.device('cuda')
        lay_drt = lay_drt.to(dev)
        inp = inp.to(dev)
        out = lay_drt(inp)
        print(out['out'].device)
        print(out)

        lay_drt = LayDRT(
            in_width=       in_width,
            do_scaled_dns=  True,
            lay_dropout=    0.1,
            res_dropout=    0.1,
            device=         dev,
            dtype=          torch.double)
        print(lay_drt)
        inp = inp.to(torch.double)
        out = lay_drt(inp)
        print(out['out'].dtype)
        print(out)

        lay_drt = LayDRT(
            in_width=       in_width,
            do_scaled_dns=  True,
            lay_dropout=    0.1,
            res_dropout=    0.1,
            dtype=          torch.double)
        self.assertRaises(RuntimeError, lay_drt, inp) # devices mismatch

    def test_EncDRT(self):

        in_width = 10
        inp = torch.rand(in_width) - 0.5
        print(inp, inp.dtype)

        enc_drt = EncDRT(in_width, shared_lays=True, dns_scale=4)
        print(enc_drt)
        self.assertTrue(len(list(enc_drt.children())) == 2) # ln_in, shared LayDRT
        out = enc_drt(inp)
        print(out)
        self.assertTrue(out['zsL'][0].shape[-1] == in_width*4)

        enc_drt = EncDRT(in_width, lay_width=2*in_width, dns_scale=3)
        print(enc_drt)
        self.assertTrue(len(list(enc_drt.children())) == 8) # projection, ln_in, 6 * LayDRT
        out = enc_drt(inp)
        print(out)
        self.assertTrue(out['zsL'][0].shape[-1] == in_width*3*2)

        enc_drt = EncDRT(
            in_width=       in_width,
            in_dropout=     0.1,
            n_layers=       4,
            lay_width=      16,
            do_scaled_dns=  True,
            dns_scale=      3,
            lay_dropout=    0.3,
            res_dropout=    0.3,
            dtype=          torch.double)
        print(enc_drt)
        self.assertTrue(len(list(enc_drt.children())) == 7) # in_drop, projection, ln_in, 4 * LayDRT
        inp = inp.to(torch.double)
        out = enc_drt(inp)
        print(out)
        self.assertTrue(len(out['zsL']) == 4) # 4 layers
        self.assertTrue(out['zsL'][0].shape[-1] == 16*3) # lay_width * dns_scale

        enc_drt.to(torch.float)
        self.assertRaises(RuntimeError, enc_drt, inp) # expected scalar type Double but found Float

    def test_EncCNN(self):

        in_features = 12
        inp = torch.rand(10,20,in_features)
        enc = EncCNN(in_features, lay_dropout=0.1)
        print(enc)
        print(enc(inp))

        #inp = torch.rand(20,in_features)
