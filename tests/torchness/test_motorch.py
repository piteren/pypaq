import numpy as np
import torch
from torch import nn
import unittest

from pypaq.torchness.motorch import MOTorch, Module, MOTorchException
from pypaq.torchness.layers import LayDense

from tests.envy import flush_tmp_dir

MOTORCH_DIR = f'{flush_tmp_dir()}/motorch'
MOTorch.SAVE_TOPDIR = MOTORCH_DIR


class LinModel(Module):

    def __init__(
            self,
            in_drop: float,
            in_shape=   784,
            out_shape=  10,
            loss_func=  torch.nn.functional.cross_entropy,
            device=     None,
            dtype=      None,
            logger=     None,
            seed=       121):
        nn.Module.__init__(self)
        self.in_drop_lay = torch.nn.Dropout(p=in_drop) if in_drop>0 else None
        self.lin = LayDense(in_features=in_shape, out_features=out_shape)
        self.loss_func = loss_func
        if logger: logger.debug('LinModel initialized!')

    def forward(self, inp) -> dict:
        if self.in_drop_lay is not None: inp = self.in_drop_lay(inp)
        logits = self.lin(inp)
        return {'logits': logits}

    def loss(self, inp, lbl) -> dict:
        out = self(inp)
        out['loss'] = self.loss_func(out['logits'], lbl)
        out['acc'] = self.accuracy(out['logits'], lbl)  # using baseline
        return out


class TestMOTorch(unittest.TestCase):

    def setUp(self) -> None:
        flush_tmp_dir()


    def test_base_init(self):
        MOTorch(
            module_type=    LinModel,
            in_drop=        0.0,
            loglevel=       10)


    def test_init_raises(self):
        self.assertRaises(Exception, MOTorch)
        kwargs = dict(name='LinModel')
        self.assertRaises(Exception, MOTorch, **kwargs)
        kwargs = dict(module_type=LinModel)
        self.assertRaises(Exception, MOTorch, **kwargs)


    def test_device(self):
        model = MOTorch(
            module_type=    LinModel,
            device=         0,
            in_drop=        0.0)
        dev = model.device
        print(dev)
        self.assertTrue(dev == 'cuda:0' if torch.cuda.is_available() else 'cpu')


    def test_save_load(self):
        model = MOTorch(
            module_type=    LinModel,
            loglevel=       10,
            in_drop=        0.1)
        self.assertTrue(model['seed']==121 and model['baseLR']==0.0003)
        self.assertTrue('loss' not in model.get_managed_params())
        model.save()
        name = model.name

        print('\nsaved, now loading..')

        model = MOTorch(
            name=       name,
            loglevel=   10)
        self.assertTrue(model['in_drop']==0.1 and model['in_shape']==784)


    def test_base_creation_and_call(self):

        model = MOTorch(
            module_type=    LinModel,
            loglevel=       10,
            in_drop=        0.1)

        inp = np.random.random((5,784)).astype(np.float32)
        lbl = np.random.randint(0,9,5)

        out = model(inp)
        print(out)
        logits = out['logits']
        self.assertTrue(logits.shape[0]==5 and logits.shape[1]==10)

        out = model.loss(inp, lbl)
        loss = out['loss']
        acc = out['acc']
        print(loss, acc)
        self.assertTrue(type(loss) is torch.Tensor)
        self.assertTrue(type(acc) is float)

        for _ in range(5):
            out = model.backward(inp, lbl)
            loss = out['loss']
            acc = out['acc']
            print(loss, acc)


    def test_class_method(self):

        model = MOTorch(
            module_type=    LinModel,
            in_drop=    0.1)
        print(model.name)
        point_org = model.get_point()
        print(point_org)
        model.save()
        point = MOTorch.load_point(name=model.name)
        print(point)
        self.assertTrue(point_org == point)


    def test_inherited_interfaces(self):

        model = MOTorch(
            module_type=    LinModel,
            in_drop=    0.1)
        print(model.parameters())
        model.float()
        print(model.state_dict())
        print(model.get_point())


    def test_training_mode(self):

        model = MOTorch(
            module_type=    LinModel,
            in_drop=    0.8)
        print(model.training)

        inp = np.random.random((5,784)).astype(np.float32)
        lbl = np.random.randint(0,9,5)

        out = model(inp)
        print(out)
        out = model.loss(inp, lbl)
        loss = out['loss']
        print(loss)

        out = model(inp, set_training=True)
        print(out)
        out = model.loss(inp, lbl, set_training=True)
        loss = out['loss']
        print(loss)


    def test_name_stamp(self):

        model = MOTorch(
            module_type=    LinModel,
            in_drop=        0.1)
        print(model['name'])
        self.assertTrue(model['name'] == 'LinModel_MOTorch')

        model = MOTorch(
            module_type=    LinModel,
            name=           'LinTest',
            in_drop=        0.1)
        print(model['name'])
        self.assertTrue(model['name'] == 'LinTest')

        model = MOTorch(
            module_type=        LinModel,
            name_timestamp=     True,
            in_drop=            0.1)
        print(model['name'])
        self.assertTrue(model['name'] != 'MOTorch_LinModel')
        self.assertTrue({d for d in '0123456789'} & set([l for l in model['name']]))


    def test_ParaSave(self):

        model = MOTorch(
            module_type=    LinModel,
            loglevel=       20,
            in_shape=       12,
            out_shape=      12,
            in_drop=        0.0)
        pms = model.get_managed_params()
        print(f'model.get_managed_params(): {pms}')
        orig_seed = model['seed']
        print(f'orig_seed: {orig_seed}')
        model.save()

        MOTorch.oversave(
            name=   model["name"],
            seed=   252)

        # this will not load
        dna = model.load_point(
            name=           model["name"],
            save_topdir=    'other')
        print(dna)
        self.assertFalse(dna)

        # this will load
        dna = model.load_point(name=model["name"])
        print(dna)
        for p in pms:
            if p not in dna: print(p)
            self.assertTrue(p in dna)
        self.assertTrue(dna["seed"]==252)

        # this model will not load
        model = MOTorch(
            module_type=    LinModel,
            save_topdir=    '_models',
            in_drop=        0.0)
        print(f'not loaded model in_shape: {model["in_shape"]}')
        self.assertTrue(model['in_shape'] != 12)

        # this model will load from MOTORCH_DIR
        model = MOTorch(
            module_type=    LinModel,
            loglevel=       10)
        print(model['in_shape'])
        self.assertTrue(model['in_shape'] == 12)

        model = MOTorch(
            module_type=    LinModel,
            name_timestamp= True,
            in_shape=       12,
            out_shape=      12,
            in_drop=        0.0)
        model.save()
        print(model['name'])

        model = MOTorch(name=model['name'])
        self.assertTrue(model['in_shape'] == 12)

        model.copy_saved_point(
            name_src=           model["name"],
            name_trg=           'CopiedPS')

        model.gx_saved_point(
            name_parent_main=           model["name"],
            name_parent_scnd=           None,
            name_child=                 'GXed')

        psdd = {'seed': [0,1000]}
        model = MOTorch(
            module_type=    LinModel,
            name=       'GXLin',
            psdd=       psdd,
            in_drop=    0.0)

        print(model.gxable)
        print(model['psdd'])
        print(model['seed'])
        dna = model.gx_point(
            parent_main=    model,
            prob_noise=     0.0,
            prob_axis=      0.0)
        print(dna['seed'])
        dna = model.gx_point(
            parent_main=    model,
            prob_noise=     1.0,
            prob_axis=      1.0)
        print(dna['seed'])


    def test_params_resolution(self):

        model = MOTorch(
            module_type=    LinModel,
            in_drop=    0.1)
        print(model['seed'])        # value from MOTORCH_DEFAULTS
        print(model['in_shape'])    # value from module_type defaults
        self.assertTrue(model['seed'] == 121)
        self.assertTrue(model['in_shape'] == 784)

        model = MOTorch(
            module_type=    LinModel,
            seed=       151,
            in_drop=    0.1)
        print(model['seed'])        # MOTORCH_DEFAULTS overridden with kwargs
        self.assertTrue(model['seed'] == 151)

        model = MOTorch(
            module_type=    LinModel,
            in_shape=   24,
            out_shape=  24,
            in_drop=    0.1)
        print(model['seed'])        # MOTORCH_DEFAULTS overridden with module_type defaults
        self.assertTrue(model['seed'] == 121)
        model.save()
        name = model.name

        model = MOTorch(
            name=   name,
            seed=   212)
        print(model['in_shape'])    # loaded from save
        print(model['seed'])        # saved overridden with kwargs
        self.assertTrue(model['in_shape'] == 24)
        self.assertTrue(model['seed'] == 212)


    def test_seed_of_torch(self):

        model = MOTorch(
            module_type=    LinModel,
            seed=       121,
            in_drop=    0.1)

        inp = np.random.random((5,784)).astype(np.float32)
        out1 = model(inp)
        print(model['seed'])
        print(out1)

        model = MOTorch(
            module_type=    LinModel,
            seed=       121,
            in_drop=    0.1)

        out2 = model(inp)
        print(model['seed'])
        print(out2)

        self.assertTrue(np.sum(out1['logits'].cpu().detach().numpy()) == np.sum(out2['logits'].cpu().detach().numpy()))


    def test_read_only(self):

        model = MOTorch(
            module_type=    LinModel,
            in_drop=    0.1)
        model.save()
        name = model.name

        model = MOTorch(
            name=       name,
            read_only=  True)
        self.assertRaises(MOTorchException, model.save)


    def test_save_load_full(self):

        model = MOTorch(
            module_type=    LinModel,
            in_shape=       256,
            out_shape=      10,
            name_timestamp= True,
            seed=           121,
            in_drop=        0.1)
        name = model.name
        print(model.name)

        inp = np.random.random((5, 256)).astype(np.float32)

        out1 = model(inp)
        print(out1)
        model.save()

        loaded_model = MOTorch(
            name=           name,
            seed=           123) # although different seed, model will load checkpoint
        print(loaded_model.name)
        out2 = loaded_model(inp)
        print(out2)
        # print(loaded_model)

        self.assertTrue(np.sum(out1['logits'].cpu().detach().numpy()) == np.sum(out2['logits'].cpu().detach().numpy()))


    def test_copy_saved(self):

        model = MOTorch(
            module_type=        LinModel,
            in_shape=       256,
            out_shape=      10,
            name_timestamp= True,
            seed=           121,
            in_drop=        0.1)
        name = model.name
        print(model)
        model.save()

        name_copied = f'{name}_copied'
        MOTorch.copy_saved(
            name_src=           name,
            name_trg=           name_copied)

        model = MOTorch(name=name_copied)
        print(model)


    def test_gx_ckpt(self):

        model = MOTorch(
            module_type=        LinModel,
            name_timestamp= True,
            seed=           121,
            in_drop=        0.1)
        name_A = model.name
        model.save()

        model = MOTorch(
            module_type=        LinModel,
            name_timestamp= True,
            seed=           121,
            in_drop=        0.1)
        name_B = model.name
        model.save()

        MOTorch.gx_ckpt(
            name_A=         name_A,
            name_B=         name_B,
            name_child=     f'{name_A}_GXed')


    def test_gx_saved(self):

        model = MOTorch(
            module_type=        LinModel,
            name_timestamp= True,
            seed=           121,
            in_drop=        0.1)
        name_A = model.name
        model.save()

        model = MOTorch(
            module_type=        LinModel,
            name_timestamp= True,
            seed=           121,
            in_drop=        0.1)
        name_B = model.name
        model.save()

        MOTorch.gx_saved(
            name_parent_main=           name_A,
            name_parent_scnd=           name_B,
            name_child=                 f'{name_A}_GXed')


    def test_hpmser_mode(self):

        model = MOTorch(
            module_type=        LinModel,
            hpmser_mode=    True,
            in_drop=        0.1)
        self.assertRaises(MOTorchException, model.save)