"""
    2022 (c) piteren

    MOTorch wraps PyTorch neural network (Module) and adds some features.

    MOTorch:
    - Builds given Module.
    - Manages one MOTorch folder (subfolder of SAVE_TOPDIR named with MOTorch name)
      for all MOTorch data (logs, params, checkpoints). MOTorch supports
      serialization into this folder.
    - Extends ParaSave, manages all init parameters. Properly resolves parameters
      using all possible sources, saves and loads them from MOTorch folder.
    - Parameters are kept in self as a Subscriptable to be easily accessed.
    - Properly resolves and holds name of object, adds stamp if needed.
    - Supports / creates logger.
    - May be read only (prevents save over).
    - May be called (with __call__) <- runs NN FWD with given data
    - May be called BWD with backward() <- runs gradient backprop for given data
    - Supports hpmser mode.
    - Manages seed and guarantees reproducibility.
    - Manages GPU / CPU device used by NN.
    - Adds TensorBoard support.
    - Defines / implements save / load / copy of whole MOTorch (ParaSave + NN checkpoint).
    - Defines interface of baseline training & testing with data loaded to Batcher.
    - Defines / implements GX.
    - Adds some sanity checks.

    Module:
        - should implement forward() and loss() methods
            - arguments for parameters of forward() will be cast to TNS (by default) by MOTorch,
        - device is managed by MOTorch with dev_pypaq: DevicesPypaq parameter

    MOTorch extends Module. By default, after init, MOTorch is set to train.mode=False.
    MOTorch manages its train.mode by itself.
"""

import numpy as np
import shutil
from sklearn.metrics import f1_score
import torch
from typing import Optional, Dict, Tuple, Any

from pypaq.lipytools.printout import stamp
from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.lipytools.moving_average import MovAvg
from pypaq.pms.base import get_params, point_trim
from pypaq.pms.parasave import ParaSave
from pypaq.mpython.devices import get_devices
from pypaq.torchness.comoneural.batcher import Batcher
from pypaq.torchness.types import TNS, DTNS
from pypaq.torchness.base_elements import mrg_ckpts
from pypaq.torchness.scaled_LR import ScaledLR
from pypaq.torchness.grad_clipping import GradClipperAVT
from pypaq.torchness.tbwr import TBwr



class MOTorchException(Exception):
    pass


# torch.nn.Module to be implemented
# forward & loss methods are needed by MOTorch.run_train()
class Module(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

    # returned DTNS should have at least 'logits' key with logits tensor for proper MOTorch.run_train()
    def forward(self, *args, **kwargs) -> DTNS:
        """
            exemplary implementation:
        return {'logits': self.logits(input)}
        """
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
            average=    'weighted', # mean weighted by support
    ) -> float:
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=-1)
        labels = labels.cpu().numpy()
        return f1_score(
            y_true=         labels,
            y_pred=         preds,
            average=        average,
            labels=         np.unique(preds),
            zero_division=  0)

    # returned DTNS should be: forward() DTNS updated with loss (and optional acc, f1)
    def loss(self, *args, **kwargs) -> DTNS:
        """
            exemplary implementation:
        out = self(input)                                                                   <- forward DTNS
        logits = out['logits']
        out['loss'] = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')   <- update with loss
        out['acc'] = self.accuracy(logits, labels)                                          <- update with acc
        out['f1'] = self.f1(logits, labels)                                                 <- update with f1
        """
        raise NotImplementedError


# extends Module (torch.nn.Module) with ParaSave and many others
class MOTorch(ParaSave, torch.nn.Module):

    SPEC_KEYS: set = {
        'loss',         # loss
        'acc',          # accuracy
        'f1'}           # F1

    MOTORCH_DEFAULTS = {
        'seed':             123,                # seed for torch and numpy
        'device':           -1,                 # :DevicesPypaq (check pypaq.mpython.devices)
        'dtype':            torch.float32,      # dtype of floats in MOTorch (16/32/64 etc)
        'bypass_data_conv': False,              # to bypass input data conversion with forward(), loss(), backward()
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
        'try_load_ckpt':    True,               # tries to load a checkpoint while init
        'hpmser_mode':      False,              # it will set model to be read_only and quiet when running with hpmser
        'read_only':        False,              # sets MOTorch to be read only - won't save anything (won't even create self.motorch_dir)
        'do_TB':            True,               # runs TensorBard, saves in self.motorch_dir
    }

    SAVE_TOPDIR = '_models'
    SAVE_FN_PFX = 'motorch_point' # POINT file prefix

    def __init__(
            self,
            module_type: Optional[type(Module)]=    None,   # also accepts torch.nn.Module but then some methods won't work (run_train, etc.)
            name: Optional[str]=                    None,
            name_timestamp=                         False,
            save_topdir: Optional[str]=             None,
            save_fn_pfx: Optional[str]=             None,
            tbwr: Optional[TBwr]=                   None,
            logger=                                 None,
            loglevel=                               20,
            **kwargs):

        if not name and not module_type:
            raise MOTorchException('name OR module_type must be given!')

        # TODO: temporary, delete later
        if 'devices' in kwargs:
            raise MOTorchException('\'devices\' param is no more supported by MOTorch, please use device')

        self.module_type = module_type # INFO: here we save TYPE

        # generate name
        name = f'{self.module_type.__name__}_MOTorch' if not name else name
        if name_timestamp: name += f'_{stamp()}'
        self.name = name

        # some early overrides

        if kwargs.get('hpmser_mode', False):
            loglevel = 50
            kwargs['read_only'] = True

        if kwargs.get('read_only', False):
            kwargs['do_TB'] = False

        _read_only = kwargs.get('read_only', False)

        if not save_topdir: save_topdir = self.SAVE_TOPDIR
        if not save_fn_pfx: save_fn_pfx = self.SAVE_FN_PFX

        if not logger:
            logger = get_pylogger(
                name=       self.name,
                add_stamp=  False,
                folder=     None if _read_only else MOTorch.__get_model_dir(save_topdir, self.name),
                level=      loglevel)
        self._log = logger

        mod_info = self.module_type.__name__ if self.module_type else 'module_type NOT GIVEN (will try to load from saved)'
        self._log.info(f'*** MOTorch : {self.name} *** initializes for module_type: {mod_info}')
        self._log.info(f'> {self.name} save_topdir: {save_topdir}{" <- read only mode!" if _read_only else ""}')

        # ************************************************************************************************* manage point

        # load point from folder
        point_saved = ParaSave.load_point(
            name=           self.name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx)

        # in case 'module_type' was not given with init, try to get it from saved
        if not self.module_type:
            self.module_type = point_saved.get('module_type', None)

        if not self.module_type:
            msg = 'module_type was not given and has not been found in saved, cannot continue!'
            self._log.error(msg)
            raise MOTorchException(msg)

        # get defaults of Module.__init__ method
        _init_method_params = get_params(self.module_type.__init__)
        _init_method_params_defaults = _init_method_params['with_defaults']   # get init params defaults

        _init_method_params_defaults_for_update = {}
        _init_method_params_defaults_for_update.update(_init_method_params_defaults)

        # we do not want params below with None to set MOTorch
        for param in ['device','dtype','logger']:
            if param in _init_method_params_defaults_for_update:
                if _init_method_params_defaults_for_update[param] is None:
                    _init_method_params_defaults_for_update.pop(param)

        # update in proper order
        self._point = {}
        self._point.update(ParaSave.PARASAVE_DEFAULTS)
        self._point.update(MOTorch.MOTORCH_DEFAULTS)
        self._point.update(_init_method_params_defaults_for_update)
        self._point.update(point_saved)
        self._point.update(kwargs)  # update with kwargs given NOW by user

        # resolve device (do it here to update 'device' param)
        # INFO: device is a special parameter, MOTorch allows it to be in DevicesPypaq type - it needs to be casted to torch namespace
        self._log.debug(f'> {self.name} resolves devices, given: {self._point["device"]}')
        self._log.debug(f'>> torch.cuda.is_available(): {torch.cuda.is_available()}')
        device = get_devices(
            devices=            self._point["device"],
            torch_namespace=    True,
            logger=             get_child(self._log, 'get_devices'))[0]
        self._log.info(f'> {self.name} given devices: {self._point["device"]}, will use: {device}')

        self._point.update({
            'name':         self.name,
            'save_topdir':  save_topdir,
            'save_fn_pfx':  save_fn_pfx,
            'device':       device})

        _point_module = {}
        _point_module.update(self._point)
        _point_module['logger'] = self._log # INFO: we need to set logger like this, since _log is not in managed_params
        self._point_module = point_trim(self.module_type.__init__, _point_module)

        not_used_kwargs = {}
        for k in kwargs:
            if k not in self._point_module:
                not_used_kwargs[k] = kwargs[k]

        self._log.debug(f'> {self.name} POINT sources:')
        self._log.debug(f'>> PARASAVE_DEFAULTS:         {ParaSave.PARASAVE_DEFAULTS}')
        self._log.debug(f'>> MOTORCH_DEFAULTS:          {MOTorch.MOTORCH_DEFAULTS}')
        self._log.debug(f'>> Module defaults:           {_init_method_params_defaults}')
        self._log.debug(f'>> POINT saved:               {point_saved}')
        self._log.debug(f'>> given kwargs:              {kwargs}')
        self._log.debug(f'> resolved POINT:')
        self._log.debug(f'Module complete POINT:        {self._point_module}')
        self._log.debug(f'>> kwargs not used by Module: {not_used_kwargs}')
        self._log.debug(f'{self.name} complete POINT:\n{self._point}')

        # ************************************************************************************************ init ParaSave

        parasave_logger = get_child(self._log, name='parasave')
        ParaSave.__init__(self, logger=parasave_logger, **self._point)

        # params names safety check
        pms = sorted(list(self.SPEC_KEYS) + list(MOTorch.MOTORCH_DEFAULTS.keys()) + list(kwargs.keys()))
        found = self.check_params_sim(params=pms)
        if found:
            self._log.warning('MOTorch was asked to check for params similarity and found:')
            for pa, pb in found: self._log.warning(f'> params \'{pa}\' and \'{pb}\' are close !!!')

        # ************************set seed in all possible areas (https://pytorch.org/docs/stable/notes/randomness.html)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # ***************************************************************************************** build MOTorch Module

        self._log.info(f'{self.name} builds graph')
        torch.nn.Module.__init__(self) # init self as a torch.nn.Module
        self._module = self.module_type(**self._point_module) # private not to be saved with point

        if self.try_load_ckpt:
            self.load_ckpt() # TODO do we want to do sth with returned additional data?
        else:
            self._log.info(f'> {self.name} checkpoint not loaded, not even tried because \'try_load_ckpt\' was set to {self.try_load_ckpt}')

        self._log.debug(f'> moving {self.name} to device: {self.device}, dtype: {self.dtype}')
        self.to(self.device)
        self.to(self.dtype)

        self._log.debug(f'{self.name} module initialized!')

        # optimizer params may be given with 'opt_' prefix
        opt_params = {k[4:]: self[k] for k in self.get_managed_params() if k.startswith('opt_')}
        opt_params.pop('class')
        self._opt = self.opt_class(
            params= self.parameters(),
            lr=     self.baseLR,
            **opt_params)
        #print(len(self._opt.param_groups))
        #print(self._opt.param_groups[0].keys())

        # from now LR is managed by scheduler
        self._scheduler = ScaledLR(
            optimizer=      self._opt,
            starting_step=  self.train_batch_IX,
            warm_up=        self.warm_up,
            ann_base=       self.ann_base,
            ann_step=       self.ann_step,
            n_wup_off=      self.n_wup_off,
            logger=         get_child(self._log, 'ScaledLR'))

        self._grad_clipper = GradClipperAVT(
            module=         self,
            clip_value=     self.clip_value,
            avt_SVal=       self.avt_SVal,
            avt_window=     self.avt_window,
            avt_max_upd=    self.avt_max_upd,
            do_clip=        self.do_clip,
            logger=         get_child(self._log, 'GradClipperAVT'))

        # MOTorch by default is not in training mode
        self.train(False)
        self._log.debug(f'> set {self.name} train.mode to False..')

        # *********************************************************************************************** other & finish

        self._TBwr = tbwr or TBwr(logdir=MOTorch.__get_model_dir(self.save_topdir, self.name)) if self.do_TB else None  # TensorBoard writer

        self._batcher = None

        self._log.debug(str(self))
        self._log.info(f'MOTorch init finished!')

    # **************************************************************************** model call (run NN with data) methods

    # forward call on NN
    def __call__(self, *args, **kwargs) -> dict:
        return torch.nn.Module.__call__(self, *args, **kwargs)

    # converts given data to torch.Tensor compatible with self (type,device,dtype)
    def convert(self, data:Any) -> TNS:

        # do not convert None
        if type(data) is not None:

            if type(data) is not torch.Tensor:
                if type(data) is np.ndarray: data = torch.from_numpy(data)
                else:                        data = torch.tensor(data)

            # convert device + float types
            data = data.to(self.device, self.dtype if data.is_floating_point() or data.is_complex() else None)

        return data

    # runs forward on nn.Module (with current nn.Module.training.mode - by default not training)
    def forward(
            self,
            *args,
            bypass_data_conv=               False,
            set_training: Optional[bool]=   None,
            **kwargs) -> DTNS:
        """
        INFO: since MOTorch is a torch.nn.Module, call forward() call should be avoided,
        INFO: instead use just MOTorch.__call__() /self()
        """

        if set_training is not None: self.train(set_training)

        if not (bypass_data_conv or self.bypass_data_conv):
            args = [self.convert(data=a) for a in args]
            kwargs = {k: self.convert(data=kwargs[k]) for k in kwargs}

        out = self.module(*args, **kwargs)

        if set_training: self.train(False) # eventually roll back to default
        return out

    # forward + loss call on NN (with current nn.Module.training.mode - by default not training)
    def loss(
            self,
            *args,
            bypass_data_conv=               False,
            set_training: Optional[bool]=   None,   # for not None forces given training mode for torch.nn.Module
            **kwargs) -> DTNS:

        if set_training is not None: self.train(set_training)

        if not (bypass_data_conv or self.bypass_data_conv):
            args = [self.convert(data=a) for a in args]
            kwargs = {k: self.convert(data=kwargs[k]) for k in kwargs}

        out = self.module.loss(*args, **kwargs)

        if set_training: self.train(False) # eventually roll back to default
        return out

    # backward call on NN, runs loss calculation + update of nn.Module (by default with training.mode = True)
    def backward(
            self,
            *args,
            bypass_data_conv=   False,
            set_training: bool= True, # for backward training mode is set to True by default
            empty_cuda_cache=   True,
            **kwargs) -> DTNS:

        out = self.loss(
            *args,
            bypass_data_conv=   bypass_data_conv,
            set_training=       set_training,
            **kwargs)

        out['loss'].backward()                                      # update gradients
        gnD = self._grad_clipper.clip()                             # clip gradients, adds: 'gg_norm' & 'gg_avt_norm' to out
        self._opt.step()                                            # apply optimizer
        self._opt.zero_grad()                                       # clear gradients
        self._scheduler.step()                                      # apply LR scheduler

        # releases all unoccupied cached memory currently held by the caching allocator
        if empty_cuda_cache:
            torch.cuda.empty_cache()

        out['currentLR'] = self._scheduler.get_last_lr()[0]         # INFO: we take currentLR of first group
        out.update(gnD)

        return out

    # *********************************************************************************************** load / save / copy

    @staticmethod
    def __get_model_dir(save_topdir:str, model_name:str) -> str:
        return f'{save_topdir}/{model_name}'

    # returns path of checkpoint pickle file
    @staticmethod
    def __get_ckpt_path(save_topdir:str, model_name:str) -> str:
        return f'{MOTorch.__get_model_dir(save_topdir, model_name)}/{model_name}.pt'

    # tries to load checkpoint and return additional data
    def load_ckpt(
            self,
            save_topdir: Optional[str]= None,  # allows to load from custom save_topdir
            name: Optional[str]=        None,  # allows to load custom name (model_name)
    ) -> Optional[dict]:

        ckpt_path = MOTorch.__get_ckpt_path(
            save_topdir=    save_topdir or self.save_topdir,
            model_name=     name or self.name)

        save_obj = None

        try:
            # INFO: immediately place all tensors to current device (not previously saved one)
            save_obj = torch.load(f=ckpt_path, map_location=self.device)
            self.load_state_dict(save_obj.pop('model_state_dict'))
            self._log.info(f'> {self.name} checkpoint loaded from {ckpt_path}')
        except Exception as e:
            self._log.info(f'> {self.name} checkpoint NOT loaded because of exception: {e}')

        return save_obj

    # saves model checkpoint & optionally additional data
    def save_ckpt(
            self,
            save_topdir: Optional[str]=         None,   # allows to save in custom save_topdir
            name: Optional[str]=                None,   # allows to save under custom name (model_name)
            additional_data: Optional[Dict]=    None,   # allows to save additional
    ) -> None:

        ckpt_path = MOTorch.__get_ckpt_path(
            save_topdir=    save_topdir or self.save_topdir,
            model_name=     name or self.name)

        save_obj = {'model_state_dict': self.state_dict()}
        if additional_data: save_obj.update(additional_data)

        torch.save(obj=save_obj, f=ckpt_path)

    # saves MOTorch (ParaSave POINT and model checkpoint)
    def save(self):
        if self.read_only: raise MOTorchException('read_only MOTorch cannot be saved!')
        self.save_point()
        self.save_ckpt()
        self._log.info(f'{self.__class__.__name__} {self.name} saved to {self.save_topdir}')

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
            src=    MOTorch.__get_ckpt_path(save_topdir_src, name_src),
            dst=    MOTorch.__get_ckpt_path(save_topdir_trg, name_trg))

    # copies full MOTorch folder (POINT & checkpoints)
    @classmethod
    def copy_saved(
            cls,
            name_src: str,
            name_trg: str,
            save_topdir_src: Optional[str]= None,
            save_topdir_trg: Optional[str]= None,
            save_fn_pfx: Optional[str]=     None):

        if not save_topdir_src: save_topdir_src = cls.SAVE_TOPDIR
        if not save_fn_pfx: save_fn_pfx = cls.SAVE_FN_PFX

        if save_topdir_trg is None: save_topdir_trg = save_topdir_src

        cls.copy_saved_point(
            name_src=           name_src,
            name_trg=           name_trg,
            save_topdir_src=    save_topdir_src,
            save_topdir_trg=    save_topdir_trg,
            save_fn_pfx=        save_fn_pfx)

        cls.copy_checkpoint(
            name_src=           name_src,
            name_trg=           name_trg,
            save_topdir_src=    save_topdir_src,
            save_topdir_trg=    save_topdir_trg)

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

        prep_folder(f'{save_topdir_child}/{name_child}')

        mrg_ckpts(
            ckptA=          MOTorch.__get_ckpt_path(save_topdir_A, name_A),
            ckptB=          MOTorch.__get_ckpt_path(save_topdir_B, name_B),
            ckptM=          MOTorch.__get_ckpt_path(save_topdir_child, name_child),
            ratio=          ratio,
            noise=          noise)

    # performs GX on saved MOTorch (without even building child objects)
    @classmethod
    def gx_saved(
            cls,
            name_parent_main: str,
            name_parent_scnd: Optional[str],    # if not given makes GX only with main parent
            name_child: str,
            save_topdir_parent_main: Optional[str]= None,
            save_topdir_parent_scnd: Optional[str]= None,
            save_topdir_child: Optional[str]=       None,
            save_fn_pfx: Optional[str]=             None,
            do_gx_ckpt=                             True,
            ratio: float=                           0.5,
            noise: float=                           0.03
    ) -> None:

        if not save_topdir_parent_main: save_topdir_parent_main = cls.SAVE_TOPDIR
        if not save_fn_pfx: save_fn_pfx = cls.SAVE_FN_PFX

        cls.gx_saved_point(
            name_parent_main=           name_parent_main,
            name_parent_scnd=           name_parent_scnd,
            name_child=                 name_child,
            save_topdir_parent_main=    save_topdir_parent_main,
            save_topdir_parent_scnd=    save_topdir_parent_scnd,
            save_topdir_child=          save_topdir_child,
            save_fn_pfx=                save_fn_pfx)

        if do_gx_ckpt:
            cls.gx_ckpt(
                name_A=             name_parent_main,
                name_B=             name_parent_scnd or name_parent_main,
                name_child=         name_child,
                save_topdir_A=      save_topdir_parent_main,
                save_topdir_B=      save_topdir_parent_scnd,
                save_topdir_child=  save_topdir_child,
                ratio=              ratio,
                noise=              noise)
        else:
            cls.copy_checkpoint(
                name_src=           name_parent_main,
                name_trg=           name_child,
                save_topdir_src=    save_topdir_parent_main,
                save_topdir_trg=    save_topdir_child)


    # ***************************************************************************************************** train / test

    # converts and loads data to Batcher
    def load_data(
            self,
            data_TR: Dict[str,np.ndarray],
            data_VL: Optional[Dict[str,np.ndarray]]=    None,
            data_TS: Optional[Dict[str,np.ndarray]]=    None,
            split_VL: float=                            0.0,
            split_TS: float=                            0.0):

        data_TR = {k: self.convert(data_TR[k]) for k in data_TR}
        if data_VL:
            data_VL = {k: self.convert(data_VL[k]) for k in data_VL}
        if data_TS:
            data_TS = {k: self.convert(data_TS[k]) for k in data_TS}

        self._batcher = Batcher(
            data_TR=        data_TR,
            data_VL=        data_VL,
            data_TS=        data_TS,
            split_VL=       split_VL,
            split_TS=       split_TS,
            batch_size=     self.batch_size,
            batching_type=  'random_cov',
            logger=         get_child(self._log, 'Batcher'))

    # trains model, returns optional test score
    def run_train(
            self,
            data_TR: Dict[str,np.ndarray],  # INFO: it will also accept Dict[str,torch.Tensor] :) !
            data_VL: Optional[Dict[str,np.ndarray]]=    None,
            data_TS: Optional[Dict[str,np.ndarray]]=    None,
            split_VL: float=                            0.0,
            split_TS: float=                            0.0,
            n_batches: Optional[int]=                   None,
            test_freq=                                  100,    # number of batches between tests, model SHOULD BE tested while training
            mov_avg_factor=                             0.1,
            save_max=                                   True,   # allows to save model while training (after max test)
            use_F1=                                     True,   # uses F1 as a train/test score (not acc)
        ) -> Optional[float]:

        if data_TR:
            self.load_data(
                data_TR=    data_TR,
                data_VL=    data_VL,
                data_TS=    data_TS,
                split_VL=   split_VL,
                split_TS=   split_TS)

        if not self._batcher: raise MOTorchException(f'{self.name} has not been given data for training, use load_data()')

        self._log.info(f'{self.name} - training starts [acc / F1 / loss]')
        self._log.info(f'data sizes (TR,VL,TS) samples: {self._batcher.get_data_size()}')

        if n_batches is None: n_batches = self.n_batches  # take default
        self._log.info(f'batch size:             {self["batch_size"]}')
        self._log.info(f'train for num_batches:  {n_batches}')

        batch_IX = 0                            # this loop (local) batch counter
        tr_accL = []
        tr_f1L = []
        tr_lssL = []

        score_name = 'F1' if use_F1 else 'acc'
        ts_score_max = 0                        # test score (acc or F1) max
        ts_score_all_results = []               # test score all results
        ts_score_mav = MovAvg(mov_avg_factor)   # test score (acc or F1) moving average

        ts_bIX = [bIX for bIX in range(n_batches+1) if not bIX % test_freq] # batch indexes when test will be performed
        assert ts_bIX, 'ERR: model SHOULD BE tested while training!'
        ten_factor = int(0.1*len(ts_bIX)) # number of tests for last 10% of training
        if ten_factor < 1: ten_factor = 1 # we need at least one result
        if self.hpmser_mode: ts_bIX = ts_bIX[-ten_factor:]

        while batch_IX < n_batches:

            out = self.backward(**self._batcher.get_batch(), bypass_data_conv=True)

            loss = out['loss']
            acc = out['acc'] if 'acc' in out else None
            f1 = out['f1'] if 'f1' in out else None

            batch_IX += 1
            self.train_batch_IX += 1

            if self.do_TB:
                self.log_TB(value=loss,                 tag='tr/loss',      step=self.train_batch_IX)
                self.log_TB(value=out['gg_norm'],       tag='tr/gn',        step=self.train_batch_IX)
                self.log_TB(value=out['gg_avt_norm'],   tag='tr/gn_avt',    step=self.train_batch_IX)
                self.log_TB(value=out['currentLR'],     tag='tr/cLR',       step=self.train_batch_IX)
                if acc is not None:
                    self.log_TB(value=acc,              tag='tr/acc',       step=self.train_batch_IX)
                if f1 is not None:
                    self.log_TB(value=f1,               tag='tr/F1',       step=self.train_batch_IX)

            if acc is not None: tr_accL.append(acc)
            if f1 is not None: tr_f1L.append(f1)
            tr_lssL.append(loss)

            if batch_IX in ts_bIX:

                ts_loss, ts_acc, ts_f1 = self.run_test()

                ts_score = ts_f1 if use_F1 else ts_acc
                if ts_score is not None:
                    ts_score_all_results.append(ts_score)
                if self.do_TB:
                    if ts_loss is not None:
                        self.log_TB(value=ts_loss,                      tag='ts/loss',              step=self.train_batch_IX)
                    if ts_acc is not None:
                        self.log_TB(value=ts_acc,                       tag='ts/acc',               step=self.train_batch_IX)
                    if ts_f1 is not None:
                        self.log_TB(value=ts_f1,                        tag='ts/F1',                step=self.train_batch_IX)
                    if ts_score is not None:
                        self.log_TB(value=ts_score_mav.upd(ts_score),   tag=f'ts/{score_name}_mav', step=self.train_batch_IX)

                tr_acc_nfo = f'{100*sum(tr_accL)/test_freq:.1f}' if acc is not None else '--'
                tr_f1_nfo =  f'{100*sum(tr_f1L)/test_freq:.1f}' if f1 is not None else '--'
                tr_loss_nfo = f'{sum(tr_lssL)/test_freq:.3f}'
                ts_acc_nfo = f'{100*ts_acc:.1f}' if ts_acc is not None else '--'
                ts_f1_nfo = f'{100*ts_f1:.1f}' if ts_f1 is not None else '--'
                ts_loss_nfo = f'{ts_loss:.3f}' if ts_loss is not None else '--'
                self._log.info(f'# {self["train_batch_IX"]:5d} TR: {tr_acc_nfo} / {tr_f1_nfo} / {tr_loss_nfo} -- TS: {ts_acc_nfo} / {ts_f1_nfo} / {ts_loss_nfo}')
                tr_accL = []
                tr_f1L = []
                tr_lssL = []

                if ts_score is not None and ts_score > ts_score_max:
                    ts_score_max = ts_score
                    if not self.read_only and save_max: self.save_ckpt() # model is saved for max ts_score

        # weighted (linear ascending weight) test score for last 10% test results
        ts_score_wval = None
        if ts_score_all_results:
            ts_score_wval = 0.0
            weight = 1
            sum_weight = 0
            for tr in ts_score_all_results[-ten_factor:]:
                ts_score_wval += tr*weight
                sum_weight += weight
                weight += 1
            ts_score_wval /= sum_weight

            if self.do_TB:
                self.log_TB(value=ts_score_wval, tag=f'ts/ts_{score_name}_wval', step=self.train_batch_IX)

        self._log.info(f'### model {self.name} finished training')
        if ts_score_wval is not None:
            self._log.info(f' > test_{score_name}_max:  {ts_score_max:.4f}')
            self._log.info(f' > test_{score_name}_wval: {ts_score_wval:.4f}')

        return ts_score_wval

    # tests model, returns: optional loss (average), optional accuracy, optional F1
    # optional loss <- since there may be not TS batches
    def run_test(
            self,
            data: Optional[Dict[str,np.ndarray]]=   None,
            split_TS: float=                        1.0, # if data for test will be given above, by default MOTorch will be tested on ALL
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:

        if data:
            self.load_data(data_TR=data, split_TS=split_TS)

        if not self._batcher: raise MOTorchException(f'{self.name} has not been given data for testing, use load_data() or give it while testing!')

        batches = self._batcher.get_TS_batches()
        lossL = []
        accL = []
        f1L = []
        n_all = 0
        for batch in batches:
            out = self.loss(**batch, bypass_data_conv=True)
            n_new = len(out['logits'])
            n_all += n_new
            lossL.append(out['loss']*n_new)
            if 'acc' in out: accL.append(out['acc']*n_new)
            if 'f1' in out:  f1L.append(out['f1']*n_new)

        acc_avg = sum(accL)/n_all if accL else None
        f1_avg = sum(f1L)/n_all if f1L else None
        loss_avg = sum(lossL)/n_all if lossL else None
        return loss_avg, acc_avg, f1_avg


    # *********************************************************************************************** other / properties

    # updates scheduler baseLR of 0 group
    def update_baseLR(self, lr: float):
        self.baseLR = lr # in case model will be saved >> loaded
        self._scheduler.update_base_lr0(lr)

    @property
    def module(self):
        return self._module

    @property
    def tbwr(self):
        return self._TBwr

    # logs value to TB
    def log_TB(
            self,
            value,
            tag: str,
            step: int) -> None:
        if self.do_TB: self._TBwr.add(value=value, tag=tag, step=step)
        else: self._log.warning(f'{self.name} cannot log to TensorBoard since \'do_TB\' flag was set to False!')

    @property
    def logger(self):
        return self._log

    @property
    def size(self) -> int:
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def __str__(self):
        s = f'MOTorch: {ParaSave.__str__(self)}\n'
        s += str(self._module)
        return s