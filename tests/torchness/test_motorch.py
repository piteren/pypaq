import numpy as np
import torch
from torch import nn
from typing import Dict
import unittest

from tests.envy import flush_tmp_dir

from pypaq.torchness.motorch import MOTorch, MOTorchException
from pypaq.torchness.layers import LayDense

NEMODEL_DIR = f'{flush_tmp_dir()}/motorch'


class LinModel(nn.Module):

    def __init__(
            self,
            in_shape: tuple= (784, 10)):
        nn.Module.__init__(self)
        self.lin = LayDense(*in_shape)

    def forward(self, xb) -> Dict:
        return {'logits': self.lin(xb)}


class LinModelSeed(nn.Module):

    def __init__(
            self,
            in_shape: tuple=    (784, 10),
            seed=               111):
        nn.Module.__init__(self)
        self.lin = LayDense(*in_shape)

    def forward(self, xb) -> Dict:
        return {'logits': self.lin(xb)}


class TestMOTor(unittest.TestCase):

    def test_base_creation_call(self):

        model = MOTorch(
            model=      LinModel,
            do_logfile= False,
            verb=       1)
        print(model)

        inp = np.random.random((5, 784))
        tns = torch.tensor(inp)
        tns = tns.float()

        out = model(tns)
        print(out)
        logits = out['logits']
        self.assertTrue(logits.shape[0]==5 and logits.shape[1]==10)

    def test_name_stamp(self):

        model = MOTorch(
            model=      LinModel,
            do_logfile= False)
        self.assertTrue(model['name'] == 'LinModel')

        model = MOTorch(
            model=      LinModel,
            name=       'LinTest',
            do_logfile= False)
        self.assertTrue(model['name'] == 'LinTest')

        model = MOTorch(
            model=          LinModel,
            name_timestamp= True,
            do_logfile=     False)
        self.assertTrue(model['name'] != 'LinModel')
        self.assertTrue({d for d in '0123456789'} & set([l for l in model['name']]))

    def test_ParaSave(self):

        model = MOTorch(
            model=          LinModel,
            save_topdir=    NEMODEL_DIR,
            do_logfile=     False,
            in_shape=       (12,12))
        model.save()

        # this model will not load
        model = MOTorch(
            model=          LinModel,
            do_logfile=     False)
        print(model['in_shape'])
        self.assertTrue(model['in_shape'][0] != 12)

        # this model will load from NEMODEL_DIR
        model = MOTorch(
            model=          LinModel,
            save_topdir=    NEMODEL_DIR,
            do_logfile=     False)
        print(model['in_shape'])
        self.assertTrue(model['in_shape'][0] == 12)

        model = MOTorch(
            model=          LinModel,
            name_timestamp= True,
            save_topdir=    NEMODEL_DIR,
            do_logfile=     False,
            in_shape=       (12,12))
        model.save()
        print(model['name'])

        model = MOTorch(
            model=          LinModel,
            name=           model['name'],
            save_topdir=    NEMODEL_DIR,
            do_logfile=     False)
        self.assertTrue(model['in_shape'][0] == 12)

    def test_params_resolution(self):

        model = MOTorch(
            model=          LinModel,
            do_logfile=     False)
        print(model['seed'])        # value from MOTORCH_DEFAULTS
        print(model['in_shape'])    # value from nn.Module defaults
        self.assertTrue(model['seed'] == 123)
        self.assertTrue(model['in_shape'] == (784,10))

        model = MOTorch(
            model=          LinModel,
            do_logfile=     False,
            seed=           151)
        print(model['seed'])        # MOTORCH_DEFAULTS overridden with kwargs
        self.assertTrue(model['seed'] == 151)

        model = MOTorch(
            model=          LinModelSeed,
            do_logfile=     False,
            in_shape=       (24,24))
        print(model['seed'])        # MOTORCH_DEFAULTS overridden with nn.Module defaults
        self.assertTrue(model['seed'] == 111)
        model.save()

        model = MOTorch(
            model=          LinModelSeed,
            seed=           212,
            do_logfile=     False)
        print(model['in_shape'])    # loaded from save
        print(model['seed'])        # saved overridden with kwargs
        self.assertTrue(model['in_shape'] == (24, 24))
        self.assertTrue(model['seed'] == 212)

    def test_seed_of_torch(self):

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

    def test_read_only(self):

        model = MOTorch(
            model=          LinModel,
            do_logfile=     False)
        model.save()

        model = MOTorch(
            model=          LinModel,
            read_only=      True,
            do_logfile=     False)
        self.assertRaises(MOTorchException, model.save)

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

    def test_hpmser_mode(self):

        model = MOTorch(
            model=          LinModel,
            hpmser_mode=    True,
            do_logfile=     False,
            verb=           1)
        self.assertTrue(model['verb'] == 0)
        self.assertRaises(MOTorchException, model.save)



if __name__ == '__main__':
    unittest.main()