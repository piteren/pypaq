"""
    MOTorch

    By default, after init MOTorch is set to train.mode=False. MOTorch manages its train.mode by itself.

"""

from abc import abstractmethod, ABC
import numpy as np
import shutil
import torch
from typing import Optional, Tuple, Dict

from pypaq.lipytools.little_methods import stamp, get_params, get_func_dna, prep_folder
from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.pylogger import get_pylogger, get_hi_child
from pypaq.pms.base_types import POINT
from pypaq.pms.parasave import ParaSave
from pypaq.mpython.devices import get_devices
from pypaq.comoneural.batcher import Batcher
from pypaq.torchness.base_elements import ScaledLR, GradClipperAVT, TBwr, mrg_ckpts


# restricted keys for fwd_func DNA and return DNA (if they appear in kwargs, should be named exactly like below)
SPEC_KEYS = [
    'train_vars',   # list of variables to train (may be returned, otherwise all trainable are taken)
    'opt_vars',     # list of variables returned by opt_func
    'loss',         # loss
    'acc',          # accuracy
    'f1',           # F1
]

# defaults below may be given with MOTorch kwargs or overridden by fwd_graph attributes
MOTORCH_DEFAULTS = {
    'seed':         123,                # seed for torch and numpy
    'devices':      -1,                         # : DevicesParam (check pypaq.mpython.devices)
        # training
    'batch_size':   64,                 # training batch size
    'n_batches':    1000,               # default length of training
    'opt_class':    torch.optim.Adam,   # default optimizer
        # LR management (check pypaq.torchness.base_elements.ScaledLR)
    'baseLR':       3e-4,
    'warm_up':      None,
    'ann_base':     None,
    'ann_step':     1.0,
    'n_wup_off':    2.0,
        # gradients clipping parameters (check pypaq.torchness.base_elements.GradClipperAVT)
    'clip_value':   None,
    'avt_SVal':     0.1,
    'avt_window':   100,
    'avt_max_upd':  1.5,
    'do_clip':      False,
        # other
    'hpmser_mode':  False,              # it will set model to be read_only and quiet when running with hpmser
    'read_only':    False,              # sets model to be read only - wont save anything (wont even create self.model_dir)
    'do_logfile':   True,               # enables saving log file in self.model_dir
    'do_TB':        True}               # runs TensorBard, saves in self.model_dir


class MOTorchException(Exception):
    pass

# torch.nn.Module to be implemented with forward & loss methods
class Module(ABC, torch.nn.Module):

    # returned dict should have at least 'logits' key with logits tensor
    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict:
        raise NotImplementedError

    # baseline accuracy implementation for logits & lables
    @staticmethod
    def accuracy(
            logits: torch.Tensor,
            labels: torch.Tensor) -> float:
        logits = logits.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        pred = np.argmax(logits, axis=-1)
        return float(np.average(pred == labels))

    # returned dict updates forward() Dict with loss & acc keys (accuracy or any other (increasing) performance float)
    @abstractmethod
    def loss_acc(self, *args, **kwargs) -> Dict:
        raise NotImplementedError

# extends Module (torch.nn.Module) with ParaSave and many others
class MOTorch(ParaSave, Module):

    SAVE_TOPDIR = '_models'
    SAVE_FN_PFX = 'motorch_dna' # filename (DNA) prefix

    def __init__(
            self,
            module: type(Module),
            name: Optional[str]=    None,
            name_timestamp=         False,      # adds timestamp to the model name
            save_topdir: str=       SAVE_TOPDIR,
            save_fn_pfx: str=       SAVE_FN_PFX,
            logger=                 None,
            loglevel=               20,
            **kwargs):

        self.module = module

        self.name = self.module.__name__ if not name else name
        if name_timestamp: self.name += f'_{stamp()}'

        # ************************************************************************* manage (resolve) DNA & init ParaSave

        # load dna from folder
        dna_saved = MOTorch.load_dna(
            name=           name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx)

        _module_init_params = get_params(self.module.__init__)
        _module_init_params_defaults = _module_init_params['with_defaults']   # get init params defaults

        dna = {}
        dna.update(MOTORCH_DEFAULTS)
        dna.update(_module_init_params_defaults)
        dna.update(dna_saved)
        dna.update(kwargs)          # update with kwargs given NOW by user
        dna.update({
            'name':         self.name,
            'save_topdir':  save_topdir,
            'save_fn_pfx':  save_fn_pfx})

        # hpmser_mode -> override
        if dna['hpmser_mode']:
            loglevel = 50
            dna['read_only'] = True

        # read only -> override
        if dna['read_only']:
            dna['do_logfile'] = False
            dna['do_TB'] = False

        self.model_dir = f'{save_topdir}/{self.name}'
        if not dna['read_only']: prep_folder(self.model_dir)

        if not logger:
            logger = get_pylogger(
                name=       self.name,
                add_stamp=  not name_timestamp,
                folder=     None if dna['read_only'] else self.model_dir,
                level=      loglevel)

        self.__log = logger
        self.__log.info(f'*** MOTorch *** {self.name} (type: {type(self).__name__}) initializes..')

        ParaSave.__init__(
            self,
            lock_managed_params=    True,
            logger=                 get_hi_child(self.__log, 'ParaSave'),
            **dna)
        self.check_params_sim(params= SPEC_KEYS + list(MOTORCH_DEFAULTS.keys())) # safety check

        dna = self.get_point()
        dna_module_keys = _module_init_params['without_defaults'] + list(_module_init_params['with_defaults'].keys())
        dna_module_keys.remove('self')
        dna_module = {k: dna[k] for k in dna_module_keys}

        not_used_kwargs = {}
        for k in kwargs:
            if k not in get_func_dna(self.module.__init__, dna):
                not_used_kwargs[k] = kwargs[k]

        self.__log.debug(f'> MOTorch DNA sources:')
        self.__log.debug(f'>> MOTORCH_DEFAULTS:                  {MOTORCH_DEFAULTS}')
        self.__log.debug(f'>> Module init defaults:              {_module_init_params_defaults}')
        self.__log.debug(f'>> DNA saved:                         {dna_saved}')
        self.__log.debug(f'>> given kwargs:                      {kwargs}')
        self.__log.debug(f'>> MOTorch kwargs not used by model : {not_used_kwargs}')
        self.__log.debug(f'Module DNA:                           {dna_module}')
        self.__log.debug(f'MOTorch complete DNA:                 {dna}')

        # *********************************************************************************************** manage devices

        self.torch_dev = self.__manage_devices()

        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(self['seed'])
        torch.cuda.manual_seed(self['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.module.__init__(self, **dna_module)
        self.to(self.torch_dev)

        try:
            self.load_ckpt()
            self.__log.debug(f'> MOTorch checkpoint loaded from {self.__get_ckpt_path()}')
        except Exception as e:
            self.__log.debug(f'> MOTorch checkpoint NOT loaded ({e})..')

        self.__TBwr = TBwr(logdir=self.model_dir)  # TensorBoard writer

        self._batcher = None

        self.opt = self['opt_class'](
            params= self.parameters(),
            lr=     self['baseLR'])

        self.scheduler = ScaledLR(
            optimizer=  self.opt,
            warm_up=    self['warm_up'],
            ann_base=   self['ann_base'],
            ann_step=   self['ann_step'],
            n_wup_off=  self['n_wup_off'],
            logger=     get_hi_child(self.__log, 'ScaledLR'))

        self.grad_clipper = GradClipperAVT(
            module=         self,
            clip_value=     self['clip_value'],
            avt_SVal=       self['avt_SVal'],
            avt_window=     self['avt_window'],
            avt_max_upd=    self['avt_max_upd'],
            do_clip=        self['do_clip'],
            logger=         get_hi_child(self.__log, 'GradClipperAVT'))

        self.__set_training(False)
        self.__log.debug(f'> set MOTorch train.mode to False..')
        self.__log.info(f'MOTorch init finished!')

    # sets CPU / GPU devices for MOTorch
    def __manage_devices(self):

        self.__log.debug(f'> MOTorch resolves devices, given: {self["devices"]}, torch.cuda.is_available(): {torch.cuda.is_available()}')

        self['devices'] = get_devices(
            devices=    self['devices'],
            namespace=  'torch',
            logger=     get_hi_child(self.__log, 'get_devices'))
        self.__log.debug(f'> MOTorch will use devices: {self["devices"]}')
        # TODO: by now supported is only the first given device
        return torch.device(self['devices'][0])

    # returns path of checkpoint pickle file
    @staticmethod
    def __get_ckpt_path_static(model_dir:str, model_name:str) -> str:
        return f'{model_dir}/{model_name}.pt'

    # returns path of checkpoint pickle file
    def __get_ckpt_path(self) -> str:
        return self.__get_ckpt_path_static(self.model_dir, self.name)

    # reloads model checkpoint
    def load_ckpt(self):
        # TODO: load all that has been saved
        checkpoint = torch.load(
            self.__get_ckpt_path(),
            map_location=   self.torch_dev) # INFO: to immediately place all tensors to current device (not previously saved one)
        self.load_state_dict(checkpoint['model_state_dict'])

    # saves model checkpoint
    def save_ckpt(self):
        # TODO: decide what to save
        torch.save({
            #'epoch': 5,
            'model_state_dict': self.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            #'loss': 0.4
        }, self.__get_ckpt_path())

    # converts all values given with args & kwargs to tensors and moves to self.torch_dev (device)
    def __torch_dev(
            self,
            *args,
            to_torch=   True,  # converts given data to torch.Tensors
            to_devices= True,  # moves tensors to devices
            **kwargs):
        if to_torch:
            args = [torch.tensor(a) if type(a) is not torch.Tensor else a for a in args]
            kwargs = {k: torch.tensor(kwargs[k]) if type(kwargs[k]) is not torch.Tensor else kwargs[k] for k in kwargs}
        if to_devices:
            args = [a.to(self.torch_dev) for a in args]
            kwargs = {k: kwargs[k].to(self.torch_dev) for k in kwargs}
        return args, kwargs

    # sets self (as nn.Module) training.mode
    def __set_training(self, mode: bool):
        torch.nn.Module.train(self, mode=mode)

    # runs forward on nn.Module (with current nn.Module.training.mode - by default not training)
    # INFO: since MOTorch is a nn.Module call of forward() should be avoided, instead use just MOTorch.__call__()
    def forward(
            self,
            *args,
            to_torch=   True,                       # converts given data to torch.Tensors
            to_devices= True,                       # moves tensors to devices
            set_training: Optional[bool]=   None,   # for not None forces given training mode for torch.nn.Module
            **kwargs) -> Dict:
        if set_training is not None: self.__set_training(set_training)
        args, kwargs = self.__torch_dev(*args, to_torch=to_torch, to_devices=to_devices, **kwargs)
        out = self.module.forward(self, *args, **kwargs)
        if set_training: self.__set_training(False) # eventually roll back to default
        return out

    # runs loss calculation on nn.Module (with current nn.Module.training.mode - by default not training)
    def loss_acc(
            self,
            *args,
            to_torch=                       True,   # converts given data to torch.Tensors
            to_devices=                     True,   # moves tensors to devices
            set_training: Optional[bool]=   None,   # for not None forces given training mode for torch.nn.Module
            **kwargs) -> Dict:
        if set_training is not None: self.__set_training(set_training)
        args, kwargs = self.__torch_dev(*args, to_torch=to_torch, to_devices=to_devices, **kwargs)
        out = self.module.loss_acc(self, *args, **kwargs)
        if set_training: self.__set_training(False) # eventually roll back to default
        return out

    # runs loss calculation + update of nn.Module (by default with training.mode = True)
    def backward(
            self,
            *args,
            to_torch=           True,   # converts given data to torch.Tensors
            to_devices=         True,   # moves tensors to devices
            set_training: bool= True,
            **kwargs) -> Dict:

        out = self.loss_acc(
            *args,
            to_torch=       to_torch,
            to_devices=     to_devices,
            set_training=   set_training,
            **kwargs)

        out['loss'].backward()          # update gradients
        gnD = self.grad_clipper.clip()  # clip gradients
        self.opt.step()                 # apply optimizer
        self.opt.zero_grad()            # clear gradients
        self.scheduler.step()           # apply LR scheduler

        out['currentLR'] = self.scheduler.get_last_lr()[0] # INFO: we take currentLR of first group
        out.update(gnD)

        return out

    # logs value to TB
    def log_TB(
            self,
            value,
            tag: str,
            step: int):

        if self['do_TB']: self.__TBwr.add(value=value, tag=tag, step=step)
        else: self.__log.warn(f'NEModel {self.name} cannot log TensorBoard since do_TB flag is False!')

    # **************************************************************************************** baseline training methods

    # loads data to Batcher
    def load_data(self, data: Dict):

        if 'train' not in data: raise MOTorchException('given data should be a dict with at least "train" key present!')

        # eventually convert to torch.Tensors and move to device
        data_td = {}
        for k in data:
            data_td[k] = {}
            for l in data[k]:
                d = data[k][l]
                if type(d) is not torch.Tensor: d = torch.tensor(d)
                d = d.to(self.torch_dev)
                data_td[k][l] = d

        self._batcher = Batcher(
            data_TR=        data_td['train'],
            data_VL=        data_td['valid'] if 'valid' in data_td else None,
            data_TS=        data_td['test'] if 'test' in data_td else None,
            batch_size=     self['batch_size'],
            batching_type=  'random_cov',
            logger=         get_hi_child(self.__log, 'Batcher'))

    # trains model
    def train(
            self,
            data=                       None,
            n_batches: Optional[int]=   None,
            test_freq=                  100,    # number of batches between tests, model SHOULD BE tested while training
            mov_avg_factor=             0.1,
            save=                       True    # allows to save model while training
    ) -> float:

        if data is not None: self.load_data(data)
        if not self._batcher: raise MOTorchException('MOTorch has not been given data for training, use load_data() or give it while training!')

        self.__log.info(f'{self.name} - training starts [acc/loss]')

        self.__set_training(True)

        if n_batches is None: n_batches = self['n_batches']  # take default
        batch_IX = 0
        tr_lssL = []
        tr_accL = []
        ts_acc_max = 0                      # test accuracy max
        ts_acc_mav = MovAvg(mov_avg_factor) # test accuracy moving average

        ts_results = []
        ts_bIX = [bIX for bIX in range(n_batches+1) if not bIX % test_freq] # batch indexes when test will be performed
        assert ts_bIX, 'ERR: model SHOULD BE tested while training!'
        ten_factor = int(0.1*len(ts_bIX)) # number of tests for last 10% of training
        if ten_factor < 1: ten_factor = 1 # we need at least one result
        if self['hpmser_mode']: ts_bIX = ts_bIX[-ten_factor:]

        while batch_IX < n_batches:

            out = self.backward(
                to_torch=   False,
                to_devices= False,
                **self._batcher.get_batch())
            batch_IX += 1

            if self['do_TB']:
                self.log_TB(value=out['loss'],        tag='tr/loss',    step=batch_IX)
                self.log_TB(value=out['acc'],         tag='tr/acc',     step=batch_IX)
                self.log_TB(value=out['gg_norm'],     tag='tr/gn',      step=batch_IX)
                self.log_TB(value=out['gg_avt_norm'], tag='tr/gn_avt',  step=batch_IX)
                self.log_TB(value=out['currentLR'],   tag='tr/cLR',     step=batch_IX)
            tr_lssL.append(out['loss'])
            tr_accL.append(out['acc'])

            if batch_IX in ts_bIX:

                self.__set_training(False)
                ts_acc, ts_loss = self.test()
                self.__set_training(True)

                acc_mav = ts_acc_mav.upd(ts_acc)
                ts_results.append(ts_acc)
                if self['do_TB']:
                    self.log_TB(value=ts_loss, tag='ts/loss',    step=batch_IX)
                    self.log_TB(value=ts_acc,  tag='ts/acc',     step=batch_IX)
                    self.log_TB(value=acc_mav, tag='ts/acc_mav', step=batch_IX)
                self.__log.info(f'{batch_IX:5d} TR: {100*sum(tr_accL)/test_freq:.1f} / {sum(tr_lssL)/test_freq:.3f} -- TS: {100*ts_acc:.1f} / {ts_loss:.3f}')
                tr_lssL = []
                tr_accL = []

                if ts_acc > ts_acc_max:
                    ts_acc_max = ts_acc
                    if not self['read_only'] and save: self.save_ckpt() # model is saved for max_ts_acc

        # weighted (linear ascending weight) test value for last 10% test results
        ts_wval = 0.0
        weight = 1
        sum_weight = 0
        for tr in ts_results[-ten_factor:]:
            ts_wval += tr*weight
            sum_weight += weight
            weight += 1
        ts_wval /= sum_weight

        if self['do_TB']: self.log_TB(value=ts_wval, tag='ts/ts_wval', step=batch_IX)
        self.__log.info(f'model {self.name} finished training')
        self.__log.info(f' > test_acc_max: {ts_acc_max:.4f}')
        self.__log.info(f' > test_wval:    {ts_wval:.4f}')

        self.__set_training(False)

        return ts_wval

    # tests model, returns accuracy and loss (average)
    def test(
            self,
            data=                           None,
            set_training: Optional[bool]=   None,   # for not None sets training mode for torch.nn.Module, allows to calculate loss with training/evaluating module mode
            ) -> Tuple[float,float]:

        if data is not None: self.load_data(data)
        if not self._batcher: raise MOTorchException('MOTorch has not been given data for testing, use load_data() or give it while testing!')

        if set_training is not None: self.__set_training(set_training)

        batches = self._batcher.get_TS_batches()
        lossL = []
        accL = []
        for batch in batches:
            out = self.loss_acc(**batch)
            lossL.append(out['loss'])
            accL.append(out['acc'])

        if set_training is not None: self.__set_training(False)  # eventually roll back to default

        return sum(accL)/len(accL), sum(lossL)/len(lossL)

    # ************************************* update ParaSave functionality (mostly to override SAVE_TOPDIR & SAVE_FN_PFX)

    @staticmethod
    def load_dna(
            name: str,
            save_topdir: str=   SAVE_TOPDIR,
            save_fn_pfx: str=   SAVE_FN_PFX) -> POINT:
        return ParaSave.load_dna(
            name=           name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx)

    @staticmethod
    def oversave(
            name: str,
            save_topdir: str=   SAVE_TOPDIR,
            save_fn_pfx: str=   SAVE_FN_PFX,
            **kwargs):
        return ParaSave.oversave(
            name=           name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx,
            **kwargs)

    @staticmethod
    def copy_saved_dna(
            name_src: str,
            name_trg: str,
            save_topdir_src: str=           SAVE_TOPDIR,
            save_topdir_trg: Optional[str]= None,
            save_fn_pfx: str=               SAVE_FN_PFX):
        return ParaSave.copy_saved_dna(
            name_src=           name_src,
            name_trg=           name_trg,
            save_topdir_src=    save_topdir_src,
            save_topdir_trg=    save_topdir_trg,
            save_fn_pfx=        save_fn_pfx)

    @staticmethod
    def gx_saved_dna(
            name_parent_main: str,
            name_parent_scnd: Optional[str],
            name_child: str,
            save_topdir_parent_main: str=           SAVE_TOPDIR,
            save_topdir_parent_scnd: Optional[str]= None,
            save_topdir_child: Optional[str]=       None,
            save_fn_pfx: str=                       SAVE_FN_PFX,
    ) -> None:
        return ParaSave.gx_saved_dna(
            name_parent_main=           name_parent_main,
            name_parent_scnd=           name_parent_scnd,
            name_child=                 name_child,
            save_topdir_parent_main=    save_topdir_parent_main,
            save_topdir_parent_scnd=    save_topdir_parent_scnd,
            save_topdir_child=          save_topdir_child,
            save_fn_pfx=                save_fn_pfx)

    # copies just checkpoint
    @staticmethod
    def copy_checkpoint(
            name_src: str,
            name_trg: str,
            save_topdir_src: str=           SAVE_TOPDIR,
            save_topdir_trg: Optional[str]= None):
        if not save_topdir_trg: save_topdir_trg = save_topdir_src
        shutil.copyfile(
            src=    MOTorch.__get_ckpt_path_static(f'{save_topdir_src}/{name_src}', name_src),
            dst=    MOTorch.__get_ckpt_path_static(f'{save_topdir_trg}/{name_trg}', name_trg))

    # copies full MOTorch folder (DNA & checkpoints)
    @staticmethod
    def copy_saved(
            name_src: str,
            name_trg: str,
            save_topdir_src: str=           SAVE_TOPDIR,
            save_topdir_trg: Optional[str]= None,
            save_fn_pfx: str=               SAVE_FN_PFX):

        if save_topdir_trg is None: save_topdir_trg = save_topdir_src

        # copy DNA
        MOTorch.copy_saved_dna(
            name_src=           name_src,
            name_trg=           name_trg,
            save_topdir_src=    save_topdir_src,
            save_topdir_trg=    save_topdir_trg,
            save_fn_pfx=        save_fn_pfx)

        # copy checkpoint
        MOTorch.copy_checkpoint(
            name_src=           name_src,
            name_trg=           name_trg,
            save_topdir_src=    save_topdir_src,
            save_topdir_trg=    save_topdir_trg)

    # GX for two MOTorch checkpoints
    @staticmethod
    def gx_ckpt(
            name_A: str,                            # name parent A
            name_B: str,                            # name parent B
            name_child: str,                        # name child
            save_topdir_A: str=                 SAVE_TOPDIR,
            save_topdir_B: Optional[str]=       None,
            save_topdir_child: Optional[str]=   None,
            ratio: float=                       0.5,
            noise: float=                       0.03):

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


    # performs GX on saved MOTorch objects (MOTorch as a ParaSave and then checkpoints, without even building child objects)
    @staticmethod
    def gx_saved(
            name_parent_main: str,
            name_parent_scnd: Optional[str],                # if not given makes GX only with main parent
            name_child: str,
            save_topdir_parent_main: str=               SAVE_TOPDIR,
            save_topdir_parent_scnd: Optional[str] =    None,
            save_topdir_child: Optional[str] =          None,
            save_fn_pfx: Optional[str] =                SAVE_FN_PFX,
            do_gx_ckpt=                                 True,
            ratio: float=                               0.5,
            noise: float=                               0.03
    ) -> None:

        MOTorch.gx_saved_dna(
            name_parent_main=           name_parent_main,
            name_parent_scnd=           name_parent_scnd,
            name_child=                 name_child,
            save_topdir_parent_main=    save_topdir_parent_main,
            save_topdir_parent_scnd=    save_topdir_parent_scnd,
            save_topdir_child=          save_topdir_child,
            save_fn_pfx=                save_fn_pfx)

        if do_gx_ckpt:
            MOTorch.gx_ckpt(
                name_A=             name_parent_main,
                name_B=             name_parent_scnd or name_parent_main,
                name_child=         name_child,
                save_topdir_A=      save_topdir_parent_main,
                save_topdir_B=      save_topdir_parent_scnd,
                save_topdir_child=  save_topdir_child,
                ratio=              ratio,
                noise=              noise)
        else:
            MOTorch.copy_checkpoint(
                name_src=           name_parent_main,
                name_trg=           name_child,
                save_topdir_src=    save_topdir_parent_main,
                save_topdir_trg=    save_topdir_child)

    # saves MOTorch (ParaSave DNA and checkpoint)
    def save(self):
        if self['read_only']: raise MOTorchException('ERR: read only MOTorch cannot be saved!')
        self.save_dna()
        self.save_ckpt()
        self.__log.info(f'MOTorch {self.name} saved')

    @property
    def tbwr(self):
        return self.__TBwr

    def __str__(self):
        return f'{self.name}\n{self.module.__str__(self)}\n{ParaSave.__str__(self)}\n'