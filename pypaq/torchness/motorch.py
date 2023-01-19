"""
 2022 (c) piteren

 MOTorch implements NNWrap interface with PyTorch.

    nngraph for MOTorch:
        - should be type of Module (defined below)
        - should implement forward() and loss_acc() methods
            - ALL arguments for parameters of forward() will be cast to TNS (by default) by MOTorch,
              if not ALL are to be casted:
                - default cast should be turned off when calling MOTorch forward() or loss_acc()
                    (to_torch, to_devices, to_dtype <- False)
                - custom MOTorch forward() should override this behaviour
        - device/s are managed by MOTorch

 MOTorch extends Module. By default, after init, MOTorch is set to train.mode=False.
 MOTorch manages its train.mode by itself.
"""

import numpy as np
import shutil
from sklearn.metrics import f1_score
import torch
from typing import Optional, Tuple, Dict

from pypaq.lipytools.little_methods import stamp
from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.pylogger import get_hi_child
from pypaq.pms.parasave import ParaSave
from pypaq.comoneural.nnwrap import NNWrap, NNWrapException
from pypaq.mpython.devices import get_devices
from pypaq.torchness.types import TNS, DTNS
from pypaq.torchness.base_elements import mrg_ckpts
from pypaq.torchness.scaled_LR import ScaledLR
from pypaq.torchness.grad_clipping import GradClipperAVT


# torch.nn.Module to be implemented
# forward & loss_acc methods are needed for MOTorch.run_train()
class Module(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

    # returned dict should have at least 'logits' key with logits tensor for proper MOTorch.run_train()
    def forward(self, *args, **kwargs) -> DTNS:
        # return {'logits': self.logits(input)}
        raise NotImplementedError

    # baseline accuracy implementation for logits & lables
    def accuracy(
            self,
            logits: TNS,
            labels: TNS) -> float:
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=-1)
        labels = labels.cpu().numpy()
        return float(np.average(np.equal(preds, labels)))

    # baseline F1 implementation for logits & lables
    def f1(
            self,
            logits: TNS,
            labels: TNS,
            average=    'macro', # 'weighted'
    ) -> float:
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=-1)
        labels = labels.cpu().numpy()
        return f1_score(labels, preds, average=average, zero_division=0)

    # returned dict updates forward() Dict with loss (and optional acc, f1)
    def loss(self, *args, **kwargs) -> DTNS:
        # out = self.forward(input)
        # logits = out['logits']
        # out['loss'] = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
        # out['acc'] = self.accuracy(logits, labels)
        # out['f1'] = self.f1(logits, labels)
        raise NotImplementedError


class MOTorchException(NNWrapException):
    pass


# extends Module (torch.nn.Module) with ParaSave and many others
class MOTorch(NNWrap, torch.nn.Module):

    SPEC_KEYS = {
        'loss',         # loss
        'acc',          # accuracy
        'f1'}           # F1

    INIT_DEFAULTS = {
        'seed':             123,                # seed for torch and numpy
        'devices':          -1,                 # :DevicesParam (check pypaq.mpython.devices)
        'dtype':            torch.float32,
            # training
        'batch_size':       64,                 # training batch size
        'n_batches':        1000,               # default length of training
        'opt_class':        torch.optim.Adam,   # default optimizer
        'train_batch_IX':   0,                  # default (starting) batch index (counter)
            # LR management (check pypaq.torchness.base_elements.ScaledLR)
        'baseLR':           3e-4,
        'warm_up':          None,
        'ann_base':         None,
        'ann_step':         1.0,
        'n_wup_off':        2.0,
            # gradients clipping parameters (check pypaq.torchness.base_elements.GradClipperAVT)
        'clip_value':       None,
        'avt_SVal':         0.1,
        'avt_window':       100,
        'avt_max_upd':      1.5,
        'do_clip':          False,
            # other
        'hpmser_mode':      False,              # it will set model to be read_only and quiet when running with hpmser
        'read_only':        False,              # sets model to be read only - won't save anything (won't even create self.nnwrap_dir)
        'do_TB':            True}               # runs TensorBard, saves in self.nnwrap_dir

    SAVE_FN_PFX = 'motorch_dna' # filename (DNA) prefix

    def __init__(
            self,
            nngraph: Optional[type(Module)]=    None,   # also accepts torch.nn.Module but then some methods won't work (run_train, etc.)
            name: Optional[str]=                None,
            name_timestamp=                     False,
            save_topdir: Optional[str]=         None,
            save_fn_pfx: Optional[str]=         None,
            logger=                             None,
            loglevel=                           20,
            **kwargs):

        torch.nn.Module.__init__(self)

        self._torch_dev = None # will be set by _manage_devices()
        self._opt = None
        self._scheduler = None
        self._grad_clipper = None

        self._nngraph_module: nngraph = None

        NNWrap.__init__(
            self,
            name=           name,
            name_timestamp= name_timestamp,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx,
            logger=         logger,
            loglevel=       loglevel,
            nngraph=        nngraph,
            **kwargs)

    # ************************************************************************************************** init submethods

    def _generate_name(
            self,
            given_name: Optional[str],
            timestamp: bool) -> str:
        name = f'{self.nngraph.__name__}_MOTorch' if not given_name else given_name
        if timestamp: name += f'_{stamp()}'
        return name

    # sets CPU / GPU devices for MOTorch
    def _manage_devices(self):

        self._nwwlog.debug(f'> {self.name} resolves devices, given: {self["devices"]}, torch.cuda.is_available(): {torch.cuda.is_available()}')

        dev = get_devices(
            devices=    self['devices'],
            namespace=  'torch',
            logger=     get_hi_child(self._nwwlog, 'get_devices'))
        self._nwwlog.info(f'> {self.name} given devices: {self["devices"]}, will use {dev}')
        # TODO: by now supported is only the first given device
        self._torch_dev = torch.device(dev[0])

    # sets seed in all possible areas
    def _set_seed(self) -> None:
        # seed (https://pytorch.org/docs/stable/notes/randomness.html)
        torch.manual_seed(self['seed'])
        torch.cuda.manual_seed(self['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # builds MOTorch graph (Module)
    def _build_graph(self) -> None:

        self._nwwlog.info(f'{self.name} builds graph')
        self._nngraph_module = self.nngraph(**self._dna_nngraph)
        self.to(self._torch_dev)
        self.to(self['dtype'])
        self._nwwlog.debug(f'{self.name} (MOTorch) Module initialized!')

        try:
            self.load_ckpt()
            self._nwwlog.info(f'> {self.name} checkpoint loaded from {self.__get_ckpt_path()}')
        except Exception as e:
            self._nwwlog.info(f'> {self.name} checkpoint NOT loaded ({e})..')

        # optimizer params may be given with 'opt_' prefix
        opt_params = {k[4:]: self[k] for k in self.get_managed_params() if k.startswith('opt_')}
        opt_params.pop('class')
        self._opt = self['opt_class'](
            params= self.parameters(),
            lr=     self['baseLR'],
            **opt_params)
        #print(len(self._opt.param_groups))
        #print(self._opt.param_groups[0].keys())

        # from now LR is managed by scheduler
        self._scheduler = ScaledLR(
            optimizer=      self._opt,
            starting_step=  self['train_batch_IX'],
            warm_up=        self['warm_up'],
            ann_base=       self['ann_base'],
            ann_step=       self['ann_step'],
            n_wup_off=      self['n_wup_off'],
            logger=         get_hi_child(self._nwwlog, 'ScaledLR'))

        self._grad_clipper = GradClipperAVT(
            module=         self,
            clip_value=     self['clip_value'],
            avt_SVal=       self['avt_SVal'],
            avt_window=     self['avt_window'],
            avt_max_upd=    self['avt_max_upd'],
            do_clip=        self['do_clip'],
            logger=         get_hi_child(self._nwwlog, 'GradClipperAVT'))

        # MOTorch by default is not in training mode
        self.train(False)
        self._nwwlog.debug(f'> set {self.name} train.mode to False..')

    # **************************************************************************** model call (run NN with data) methods

    def __call__(self, *args, **kwargs) -> dict:
        return torch.nn.Module.__call__(self, *args, **kwargs)

    # converts (type,device,dtype) input data given with *args & **kwargs
    def _conv_(
            self,
            *args,
            to_torch=   True,  # converts given data to torch.Tensors
            to_devices= True,  # moves tensors to devices
            to_dtype=   True,  # converts given data to dtype
            **kwargs):
        elements = list(args) + list(kwargs.values())
        if to_torch:
            elements = [torch.tensor(e) for e in elements]
        if to_devices:
            elements = [e.to(self._torch_dev) for e in elements]
        if to_dtype:
            elements = [e.to(self['dtype']) for e in elements]
        args = elements[:len(args)]
        kwargs = {k:e for k,e in zip(kwargs.keys(), elements[len(args):])}
        return args, kwargs

    # for user managed cast of data to tensor compatible with self (type,device,dtype)
    def convert(self, data):
        return self._conv_(data)[0][0]

    # runs forward on nn.Module (with current nn.Module.training.mode - by default not training)
    # INFO: since MOTorch is a torch.nn.Module, call forward() call should be avoided, instead use just MOTorch.__call__() /self()
    def forward(
            self,
            *args,
            to_torch=                       True,
            to_devices=                     True,
            to_dtype=                       True,
            set_training: Optional[bool]=   None,   # for not None forces given training mode for torch.nn.Module
            **kwargs) -> DTNS:
        if set_training is not None: self.train(set_training)
        args, kwargs = self._conv_(*args, to_torch=to_torch, to_devices=to_devices, to_dtype=to_dtype, **kwargs)
        out = self._nngraph_module.forward(*args, **kwargs)
        if set_training: self.train(False) # eventually roll back to default
        return out

    # runs loss calculation on nn.Module (with current nn.Module.training.mode - by default not training)
    def loss(
            self,
            *args,
            # INFO: since loss() is used in training loop ( run_training() while testing ) where data is converted while loading we do not want to convert each batch separately
            to_torch=                       False,
            to_devices=                     False,
            to_dtype=                       False,
            set_training: Optional[bool]=   None,   # for not None forces given training mode for torch.nn.Module
            **kwargs) -> DTNS:
        if set_training is not None: self.train(set_training)
        args, kwargs = self._conv_(*args, to_torch=to_torch, to_devices=to_devices, to_dtype=to_dtype, **kwargs)
        out = self._nngraph_module.loss(*args, **kwargs)
        if set_training: self.train(False) # eventually roll back to default
        return out

    # runs loss calculation + update of nn.Module (by default with training.mode = True)
    def backward(
            self,
            *args,
            # INFO: since backward() is used in training loop ( run_training() ) where data is converted while loading we do not want to convert each batch separately
            to_torch=           False,
            to_devices=         False,
            to_dtype=           False,
            set_training: bool= True, # for backward training mode is set by default
            **kwargs) -> DTNS:

        out = self.loss(
            *args,
            to_torch=       to_torch,
            to_devices=     to_devices,
            to_dtype=       to_dtype,
            set_training=   set_training,
            **kwargs)

        out['loss'].backward()          # update gradients
        gnD = self._grad_clipper.clip() # clip gradients, adds: 'gg_norm' & 'gg_avt_norm' to out
        self._opt.step()                # apply optimizer
        self._opt.zero_grad()           # clear gradients
        self._scheduler.step()          # apply LR scheduler

        out['currentLR'] = self._scheduler.get_last_lr()[0] # INFO: we take currentLR of first group
        out.update(gnD)

        return out

    # *********************************************************************************************** load / save / copy

    # returns path of checkpoint pickle file
    @staticmethod
    def __get_ckpt_path_static(model_dir:str, model_name:str) -> str:
        return f'{model_dir}/{model_name}.pt'

    # returns path of checkpoint pickle file
    def __get_ckpt_path(self) -> str:
        return self.__get_ckpt_path_static(self.nnwrap_dir, self.name)

    def load_ckpt(self) -> None:
        checkpoint = torch.load(
            self.__get_ckpt_path(),
            map_location=   self._torch_dev, # INFO: to immediately place all tensors to current device (not previously saved one)
        )
        self.load_state_dict(checkpoint['model_state_dict'])

    # saves model checkpoint
    def save_ckpt(self) -> None:
        torch.save({
            #'epoch': 5,
            'model_state_dict': self.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            #'loss': 0.4
        }, self.__get_ckpt_path())

    @classmethod
    def copy_checkpoint(
            cls,
            name_src: str,
            name_trg: str,
            save_topdir_src: Optional[str]= None,
            save_topdir_trg: Optional[str]= None):
        if not save_topdir_src: save_topdir_src = cls.SAVE_TOPDIR
        if not save_topdir_trg: save_topdir_trg = save_topdir_src
        shutil.copyfile(
            src=    MOTorch.__get_ckpt_path_static(f'{save_topdir_src}/{name_src}', name_src),
            dst=    MOTorch.__get_ckpt_path_static(f'{save_topdir_trg}/{name_trg}', name_trg))

    # *************************************************************************************************************** GX

    @classmethod
    def gx_ckpt(
            cls,
            name_A: str,                            # name parent A
            name_B: str,                            # name parent B
            name_child: str,                        # name child
            save_topdir_A: Optional[str]=       None,
            save_topdir_B: Optional[str]=       None,
            save_topdir_child: Optional[str]=   None,
            ratio: float=                       0.5,
            noise: float=                       0.03):

        if not save_topdir_A: save_topdir_A = cls.SAVE_TOPDIR
        if not save_topdir_B: save_topdir_B = save_topdir_A
        if not save_topdir_child: save_topdir_child = save_topdir_A

        model_dir_child = f'{save_topdir_child}/{name_child}'
        prep_folder(model_dir_child)

        mrg_ckpts(
            ckptA=          MOTorch.__get_ckpt_path_static(f'{save_topdir_A}/{name_A}', name_A),
            ckptB=          MOTorch.__get_ckpt_path_static(f'{save_topdir_B}/{name_B}', name_B),
            ckptM=          MOTorch.__get_ckpt_path_static(model_dir_child, name_child),
            ratio=          ratio,
            noise=          noise)

    # ***************************************************************************************************** train / test

    # adds data conversion
    def load_data(
            self,
            data: Dict[str, np.ndarray],
            **kwargs):
        _, data = self._conv_(**data)
        super(MOTorch, self).load_data(data=data, **kwargs)

    # *********************************************************************************************** other / properties

    # updates scheduler baseLR of 0 group
    def update_baseLR(self, lr: float):
        self['baseLR'] = lr # in case model will be saved >> loaded
        self._scheduler.update_base_lr0(lr)

    @property
    def size(self) -> int:
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def get_devices(self):
        return self._torch_dev

    def __str__(self):
        s = f'MOTorch: {ParaSave.__str__(self)}\n'
        s += str(self._nngraph_module)
        return s