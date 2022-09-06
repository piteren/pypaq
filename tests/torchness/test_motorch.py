import numpy as np
import torch
from torch import nn
from typing import Dict
import unittest

from pypaq.torchness.todel import ToDEL
from pypaq.torchness.motorch import MOTorch


class LinToDEL(ToDEL):

    def __init__(
            self,
            in_shape: tuple=    (784, 10),
            do_logfile=         False,
            **kwargs):
        ToDEL.__init__(
            self,
            in_shape=       in_shape,
            do_logfile=     do_logfile,
            **kwargs)
        self.lin = nn.Linear(*in_shape)

    def forward(self, xb) -> Dict:
        return {'logits': self.lin(xb)}

class LinModel(nn.Module):

    def __init__(
            self,
            in_shape: tuple= (784, 10)):
        nn.Module.__init__(self)
        self.lin = nn.Linear(*in_shape)

    def forward(self, xb) -> Dict:
        return {'logits': self.lin(xb)}


class TestMOTor(unittest.TestCase):

    def test_base(self):

        model = MOTorch(model=LinModel)
        print(model)
        print(model.parameters())

        inp = np.random.random((5, 784))
        tns = torch.tensor(inp)
        tns = tns.float()

        out = model(tns)
        print(out)
        logits = out['logits']
        self.assertTrue(logits.shape[0]==5 and logits.shape[1]==10)

    def test_save_load(self):

        model = MOTorch(
            model=          LinModel,
            in_shape=       (256, 10),
            name_timestamp= True,
            verb=           1)
        model.save()
        name = model.name

        loaded_model = MOTorch(
            model=  LinModel,
            name=   name,
            verb=   1)
        print(loaded_model)


if __name__ == '__main__':
    unittest.main()