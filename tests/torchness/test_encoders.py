import torch
import unittest

from pypaq.torchness.base_elements import TorchnessException
from pypaq.torchness.encoders import LayBlockDRT, EncDRT, LayBlockCNN, EncCNN, LayBlockTNS, EncTNS


class TestEncoders(unittest.TestCase):

    def test_LayBlockDRT_base(self):

        in_width = 10
        inp = torch.rand(in_width) - 0.5
        print(inp, inp.dtype)

        lay_drt = LayBlockDRT(in_width)
        print(lay_drt)
        out = lay_drt(inp)
        print(out)
        self.assertTrue(out['out'].shape[-1] == in_width)
        self.assertTrue(out['zsL'][0].shape[-1] == in_width)


    def test_LayBlockDRT_kwargs(self):

        in_width = 10
        inp = torch.rand(in_width) - 0.5

        lay_drt = LayBlockDRT(
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


    def test_LayBlockDRT_device(self):

        in_width = 10
        inp = torch.rand(in_width) - 0.5

        lay_drt = LayBlockDRT(in_width)

        dev = torch.device('cuda')
        lay_drt = lay_drt.to(dev)
        inp = inp.to(dev)
        out = lay_drt(inp)
        print(out['out'].device)
        print(out)

        lay_drt = LayBlockDRT(in_width)
        self.assertRaises(RuntimeError, lay_drt, inp)  # devices mismatch


    def test_LayBlockDRT_double(self):

        in_width = 10
        inp = torch.rand(in_width) - 0.5

        lay_drt = LayBlockDRT(
            in_width=       in_width,
            do_scaled_dns=  True,
            lay_dropout=    0.1,
            res_dropout=    0.1,
            dtype=          torch.double)
        print(lay_drt)
        inp = inp.to(torch.double)
        out = lay_drt(inp)
        print(out['out'].dtype)
        print(out)


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


    def test_LayBlockCNN_base_encoder(self):

        n_time = 6
        n_filters = 4
        inp = torch.rand(n_time, n_filters)
        print(f'inp shape {inp.shape}')

        for kernel_size in [3,5,7,9]:
            lay_cnn = LayBlockCNN(n_filters, kernel_size=kernel_size)
            print(lay_cnn)

            out = lay_cnn(inp)
            print(f'out shape {out["out"].shape}')
            print(f'state: {out["state"]}')
            self.assertTrue(inp.shape == out['out'].shape)
            self.assertTrue(out['state'] is None)


    def test_LayBlockCNN_base_casual(self):

        n_time = 6
        n_filters = 4
        inp = torch.rand(n_time, n_filters)
        print(f'inp shape {inp.shape}')

        for kernel_size in [3,5,7,9]:
            lay_cnn = LayBlockCNN(n_filters, kernel_size=kernel_size)
            print(lay_cnn)

            zero_history_def = lay_cnn.get_zero_history()
            zero_history = lay_cnn.get_zero_history(inp)
            self.assertTrue(zero_history_def.shape == zero_history.shape)
            print(f'zero_history shape {zero_history.shape}')

            out = lay_cnn(inp, history=zero_history)
            print(f'out shape {out["out"].shape}')
            print(f'state shape: {out["state"].shape}')
            self.assertTrue(out['out'].shape == inp.shape)
            self.assertTrue(out['state'].shape == zero_history.shape)


    def test_LayBlockCNN_kwargs(self):

        n_filters = 4
        inp = torch.rand(6, n_filters)
        print(inp)

        lay_cnn = LayBlockCNN(
            n_filters=      n_filters,
            lay_dropout=    0.1,
            res_dropout=    0.1,
            do_ldrt=        True)
        print(lay_cnn)
        out = lay_cnn(inp)
        print(out)


    def test_LayBlockCNN_more(self):

        n_time = 32
        n_filters = 64
        inp = torch.rand(16, n_time, n_filters)
        print(f'inp shape {inp.shape}')

        kernel_size = 3
        lay_cnn = LayBlockCNN(n_filters, kernel_size=kernel_size)
        print(lay_cnn)

        zero_history_def = lay_cnn.get_zero_history()
        zero_history = lay_cnn.get_zero_history(inp)
        print(f'zero_history_def shape {zero_history_def.shape}')
        print(f'zero_history shape {zero_history.shape}')

        out = lay_cnn(inp, history=zero_history)
        print(f'out shape {out["out"].shape}')
        print(f'state shape: {out["state"].shape}')
        self.assertTrue(out['out'].shape == inp.shape)
        self.assertTrue(out['state'].shape == zero_history.shape)


    def test_EncCNN_base(self):

        self.assertRaises(TorchnessException, EncCNN, in_features=6, kernel_size=4) # even number for kernel

        in_features = 128
        inp = torch.rand(256,512,in_features)
        print(inp.shape)

        enc = EncCNN(in_features)
        print(enc)
        enc_out = enc(inp)

        print(enc_out['out'].shape)
        self.assertTrue(enc_out['out'].shape == inp.shape)

        self.assertTrue(enc_out['state'] is None)


    def test_EncCNN_casual(self):

        in_features = 128
        inp = torch.rand(256,512,in_features)
        print(inp.shape)

        enc = EncCNN(in_features)
        print(enc)

        zero_history_base = enc.get_zero_history()
        print(zero_history_base.shape)
        zero_history = enc.get_zero_history(inp)
        enc_out = enc(inp, history=zero_history)

        print(enc_out['out'].shape)
        self.assertTrue(enc_out['out'].shape == inp.shape)

        print(enc_out['state'].shape)
        self.assertTrue(enc_out['state'].shape == zero_history.shape)


    def test_EncCNN_kwargs(self):

        n_filters = 48
        in_features = 32
        inp = torch.rand(18,96,in_features)
        print(inp.shape)
        inp = inp.to(torch.double)

        enc = EncCNN(
            in_features=        in_features,
            time_drop=          0.1,
            feat_drop=          0.2,
            n_layers=           5,
            kernel_size=        7,
            n_filters=          n_filters,
            lay_dropout=        0.15,
            res_dropout=        0.25,
            do_ldrt=            True,
            ldrt_drop=          0.05,
            ldrt_res_dropout=   0.07,
            dtype=              torch.double)
        print(enc)
        enc_out = enc(inp)

        print(enc_out['out'].shape)
        in_sh = list(inp.shape)
        in_sh[-1] = n_filters
        self.assertTrue(list(enc_out['out'].shape) == in_sh)

        zero_history = enc.get_zero_history(inp)
        print(zero_history.shape)
        self.assertTrue(list(zero_history.shape) == [18,5,6,48])


    def test_EncCNN_shared(self):

        self.assertRaises(TorchnessException, EncCNN, in_features=6, kernel_size=4) # even number for kernel

        in_features = 128
        inp = torch.rand(256,512,in_features)
        print(inp.shape)

        enc = EncCNN(in_features, shared_lays=True)
        print(enc)
        enc_out = enc(inp)

        print(enc_out['out'].shape)
        self.assertTrue(list(enc_out['out'].shape) == list(inp.shape))

        zero_history = enc.get_zero_history(inp)
        print(zero_history.shape)
        self.assertTrue(list(zero_history.shape) == [256,6,2,128])


    def test_LayBlockTNS_base(self):

        in_features = 64
        inp = torch.rand(16,32,in_features)
        print(inp.shape)

        lay_tns = LayBlockTNS(d_model=in_features)
        print(lay_tns)
        out = lay_tns(inp)
        print(out['out'].shape)
        print(out['zsL'][0].shape)

        # TAT
        query = torch.mean(inp, dim=-2, keepdim=True)
        print(query.shape)
        out = lay_tns(inp, task_query=query)
        print(out['out'].shape)
        print(out['zsL'][0].shape)


    def test_EncTNS_base(self):

        in_features = 64
        inp = torch.rand(16,32,in_features)
        print(inp.shape)

        enc = EncTNS(num_layers=4, d_model=in_features)
        self.assertTrue(enc.pos_emb is None)
        print(enc)
        out = enc(inp)
        print(out['out'].shape, len(out['zsL']))


    def test_EncTNS_PE(self):

        in_features = 64
        seq_len = 32
        inp = torch.rand(16,seq_len,in_features)
        print(inp.shape)

        enc = EncTNS(num_layers=2, d_model=in_features, max_seq_len=48)
        print(enc)
        print(enc.pos_emb)
        self.assertTrue(enc.pos_emb is not None)
        out = enc(inp)
        print(out['out'].shape)


    def test_EncRNS_TAT(self):

        in_features = 64
        seq_len = 32
        inp = torch.rand(16, seq_len, in_features)
        print(inp.shape)

        enc = EncTNS(
            num_layers=     4,
            num_layers_TAT= 2,
            d_model=        in_features)
        print(enc)
        out = enc(inp)
        print(out['out'].shape)


    def test_EncRNS_TAT_shared(self):

        in_features = 64
        seq_len = 32
        inp = torch.rand(16, seq_len, in_features)
        print(inp.shape)

        enc = EncTNS(
            num_layers=     4,
            num_layers_TAT= 2,
            shared_lays=    (3,1,2),
            d_model=        in_features)
        print(enc)
        out = enc(inp)
        print(out['out'].shape)
