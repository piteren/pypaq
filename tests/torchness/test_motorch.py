import numpy as np
import torch
from torch import nn
from typing import Dict
import unittest

from pypaq.torchness.motorch import MOTorch
from pypaq.torchness.layers import LayDense


class LinModel(nn.Module):

    def __init__(
            self,
            in_shape: tuple= (784, 10)):
        nn.Module.__init__(self)
        self.lin = LayDense(*in_shape)

    def forward(self, xb) -> Dict:
        return {'logits': self.lin(xb)}


class TestMOTor(unittest.TestCase):

    def test_base(self):

        model = MOTorch(
            model=      LinModel,
            do_logfile= False)
        print(model)

        inp = np.random.random((5, 784))
        tns = torch.tensor(inp)
        tns = tns.float()

        out = model(tns)
        print(out)
        logits = out['logits']
        self.assertTrue(logits.shape[0]==5 and logits.shape[1]==10)

    def test_seed(self):

        inp = np.random.random((5, 784))
        tns = torch.tensor(inp)
        tns = tns.float()

        model = MOTorch(
            model=      LinModel,
            seed=       121,
            do_logfile= False)
        out1 = model(tns)
        print(model['seed'])
        print(out1)

        model = MOTorch(
            model=      LinModel,
            seed=       121,
            do_logfile= False)
        out2 = model(tns)
        print(model['seed'])
        print(out2)

        self.assertTrue(np.sum(out1['logits'].detach().numpy()) == np.sum(out2['logits'].detach().numpy()))


    def test_save_load(self):

        inp = np.random.random((5, 256))
        tns = torch.tensor(inp)
        tns = tns.float()

        model = MOTorch(
            model=          LinModel,
            in_shape=       (256, 10),
            name_timestamp= True,
            seed=           121,
            do_logfile=     False,
            verb=           0)
        name = model.name
        out1 = model(tns)
        print(out1)
        model.save()

        loaded_model = MOTorch(
            model=      LinModel,
            name=       name,
            seed=       123, # although different seed, model will load checkpoint
            do_logfile= False,
            verb=       0)
        out2 = loaded_model(tns)
        print(out2)
        # print(loaded_model)

        self.assertTrue(np.sum(out1['logits'].detach().numpy()) == np.sum(out2['logits'].detach().numpy()))


if __name__ == '__main__':
    unittest.main()