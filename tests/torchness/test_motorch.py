import numpy as np
import torch
from torch import nn
from typing import Dict
import unittest

from tests.envy import flush_tmp_dir

from pypaq.torchness.motorch import MOTorch, Module, MOTorchException
from pypaq.torchness.layers import LayDense

MOTORCH_DIR = f'{flush_tmp_dir()}/motorch'


class LinModel(Module):

    def __init__(
            self,
            in_drop=    0.0,
            in_shape=   784,
            out_shape=  10):
        nn.Module.__init__(self)
        self.in_drop_lay = torch.nn.Dropout(p=in_drop) if in_drop>0 else None
        self.lin = LayDense(in_features=in_shape, out_features=out_shape)

    def forward(self, inp) -> Dict:
        if self.in_drop_lay is not None: inp = self.in_drop_lay(inp)
        logits = self.lin(inp)
        return {'logits': logits}

    def loss(self, inp, lbl) -> torch.Tensor:
        logits = self(inp)['logits']
        loss_func = torch.nn.functional.cross_entropy
        loss = loss_func(logits, lbl)   # calculate loss
        return loss


class LinModelSeed(LinModel):

    def __init__(
            self,
            in_drop=    0.0,
            in_shape=   784,
            out_shape=  10,
            seed=       111):
        LinModel.__init__(
            self,
            in_drop=    in_drop,
            in_shape=   in_shape,
            out_shape=  out_shape)


class TestMOTor(unittest.TestCase):

    def test_base_creation_call(self):

        model = MOTorch(
            module=     LinModel,
            do_logfile= False,
            verb=       1)
        print(model)

        inp = np.random.random((5, 784))
        inp = torch.tensor(inp)
        inp = inp.float()

        lbl = np.random.randint(0,9,5)

        out = model(inp)
        print(out)
        logits = out['logits']
        self.assertTrue(logits.shape[0]==5 and logits.shape[1]==10)

        loss = model.loss(inp, lbl)
        print(loss)
        self.assertTrue(type(loss) is torch.Tensor)


    def test_inherited_interfaces(self):

        model = MOTorch(
            module=     LinModel,
            do_logfile= False,
            verb=       0)
        print(model.parameters())
        model.float()
        print(model.state_dict())
        print(model.get_point())


    def test_training_mode(self):

        model = MOTorch(
            module=     LinModel,
            in_drop=    0.8,
            do_logfile= False,
            verb=       0)
        print(model.training)

        inp = np.random.random((5, 784))
        inp = torch.tensor(inp)
        inp = inp.float()

        lbl = np.random.randint(0, 9, 5)

        out = model(inp)
        print(out)
        loss = model.loss(inp, lbl)
        print(loss)

        out = model(inp, set_training=True)
        print(out)
        loss = model.loss(inp, lbl, set_training=True)
        print(loss)


    def test_name_stamp(self):

        model = MOTorch(
            module=     LinModel,
            do_logfile= False)
        self.assertTrue(model['name'] == 'LinModel')

        model = MOTorch(
            module=     LinModel,
            name=       'LinTest',
            do_logfile= False)
        self.assertTrue(model['name'] == 'LinTest')

        model = MOTorch(
            module=         LinModel,
            name_timestamp= True,
            do_logfile=     False)
        self.assertTrue(model['name'] != 'LinModel')
        self.assertTrue({d for d in '0123456789'} & set([l for l in model['name']]))


    def test_ParaSave(self):

        model = MOTorch(
            module=         LinModel,
            save_topdir=    MOTORCH_DIR,
            do_logfile=     False,
            in_shape=       12,
            out_shape=      12,
            verb=           1)
        pms = model.get_managed_params()
        print(pms)
        model.save()

        model.oversave(
            name=           model["name"],
            save_topdir=    MOTORCH_DIR,
            seed=           252)

        # this will not load
        dna = model.load_dna(name=model["name"])
        print(dna)
        self.assertFalse(dna)

        dna = model.load_dna(
            name=           model["name"],
            save_topdir=    MOTORCH_DIR)
        print(dna)
        for p in pms:
            self.assertTrue(p in dna)

        self.assertTrue(dna["seed"]==252)

        # this model will not load
        model = MOTorch(
            module=         LinModel,
            do_logfile=     False)
        print(model['in_shape'])
        self.assertTrue(model['in_shape'] != 12)

        # this model will load from NEMODEL_DIR
        model = MOTorch(
            module=         LinModel,
            save_topdir=    MOTORCH_DIR,
            do_logfile=     False)
        print(model['in_shape'])
        self.assertTrue(model['in_shape'] == 12)

        model = MOTorch(
            module=         LinModel,
            name_timestamp= True,
            save_topdir=    MOTORCH_DIR,
            do_logfile=     False,
            in_shape=       12,
            out_shape=      12)
        model.save()
        print(model['name'])

        model = MOTorch(
            module=         LinModel,
            name=           model['name'],
            save_topdir=    MOTORCH_DIR,
            do_logfile=     False)
        self.assertTrue(model['in_shape'] == 12)

        model.copy_saved_dna(
            name_src=           model["name"],
            name_trg=           'CopiedPS',
            save_topdir_src=    MOTORCH_DIR)

        model.gx_saved_dna(
            name_parent_main=           model["name"],
            name_parent_scnd=           None,
            save_topdir_parent_main=    MOTORCH_DIR,
            name_child=                 'GXed')

        psdd = {'seed': [0,1000]}
        model = MOTorch(
            module=         LinModel,
            name=           'GXLin',
            save_topdir=    MOTORCH_DIR,
            psdd=           psdd,
            do_logfile=     False)

        print(model.gxable)
        print(model['psdd'])
        print(model['seed'])
        dna = model.gx_dna(
            parent_main=    model,
            prob_noise=     0.0,
            prob_axis=      0.0)
        print(dna['seed'])
        dna = model.gx_dna(
            parent_main=    model,
            prob_noise=     1.0,
            prob_axis=      1.0)
        print(dna['seed'])


    def test_params_resolution(self):

        model = MOTorch(
            module=         LinModel,
            do_logfile=     False)
        print(model['seed'])        # value from MOTORCH_DEFAULTS
        print(model['in_shape'])    # value from nn.Module defaults
        self.assertTrue(model['seed'] == 123)
        self.assertTrue(model['in_shape'] == 784)

        model = MOTorch(
            module=         LinModel,
            do_logfile=     False,
            seed=           151)
        print(model['seed'])        # MOTORCH_DEFAULTS overridden with kwargs
        self.assertTrue(model['seed'] == 151)

        model = MOTorch(
            module=         LinModelSeed,
            save_topdir=    MOTORCH_DIR,
            do_logfile=     False,
            in_shape=       24,
            out_shape=      24)
        print(model['seed'])        # MOTORCH_DEFAULTS overridden with nn.Module defaults
        self.assertTrue(model['seed'] == 111)
        model.save()

        model = MOTorch(
            module=         LinModelSeed,
            save_topdir=    MOTORCH_DIR,
            seed=           212,
            do_logfile=     False)
        print(model['in_shape'])    # loaded from save
        print(model['seed'])        # saved overridden with kwargs
        self.assertTrue(model['in_shape'] == 24)
        self.assertTrue(model['seed'] == 212)


    def test_seed_of_torch(self):

        inp = np.random.random((5, 784))
        tns = torch.tensor(inp)
        tns = tns.float()

        model = MOTorch(
            module=     LinModel,
            seed=       121,
            do_logfile= False)
        out1 = model(tns)
        print(model['seed'])
        print(out1)

        model = MOTorch(
            module=     LinModel,
            seed=       121,
            do_logfile= False)
        out2 = model(tns)
        print(model['seed'])
        print(out2)

        self.assertTrue(np.sum(out1['logits'].cpu().detach().numpy()) == np.sum(out2['logits'].cpu().detach().numpy()))


    def test_read_only(self):

        model = MOTorch(
            module=         LinModel,
            save_topdir=    MOTORCH_DIR,
            do_logfile=     False)
        model.save()

        model = MOTorch(
            module=         LinModel,
            read_only=      True,
            do_logfile=     False)
        self.assertRaises(MOTorchException, model.save)


    def test_save_load(self):

        inp = np.random.random((5, 256))
        tns = torch.tensor(inp)
        tns = tns.float()

        model = MOTorch(
            module=         LinModel,
            save_topdir=    MOTORCH_DIR,
            in_shape=       256,
            out_shape=      10,
            name_timestamp= True,
            seed=           121,
            do_logfile=     False,
            verb=           0)
        name = model.name
        out1 = model(tns)
        print(out1)
        model.save()

        loaded_model = MOTorch(
            module=         LinModel,
            save_topdir=    MOTORCH_DIR,
            name=           name,
            seed=           123, # although different seed, model will load checkpoint
            do_logfile=     False,
            verb=           0)
        out2 = loaded_model(tns)
        print(out2)
        # print(loaded_model)

        self.assertTrue(np.sum(out1['logits'].cpu().detach().numpy()) == np.sum(out2['logits'].cpu().detach().numpy()))


    def test_hpmser_mode(self):

        model = MOTorch(
            module=         LinModel,
            hpmser_mode=    True,
            do_logfile=     False,
            verb=           1)
        self.assertTrue(model['verb'] == 0)
        self.assertRaises(MOTorchException, model.save)


if __name__ == '__main__':
    unittest.main()