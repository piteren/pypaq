"""
 2022 (c) piteren

 MOTorch implements NNWrap interface with PyTorch.

    nngraph for MOTorch:
        - Should be type of Module (defined below).
        - Should implement forward() and loss_acc() methods.

 MOTorch extends Module. By default, after init, MOTorch is set to train.mode=False.
 MOTorch manages its train.mode by itself.
"""

from abc import abstractmethod, ABC
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


# torch.nn.Module to be implemented with forward & loss methods
class Module(ABC, torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

    # returned dict should have at least 'logits' key with logits tensor
    @abstractmethod
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
    def f1(self,
           logits: TNS,
           labels: TNS) -> float:
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=-1)
        labels = labels.cpu().numpy()
        return f1_score(labels, preds, zero_division=0)

    # returned dict updates forward() Dict with loss & acc keys (accuracy or any other (increasing) performance float)
    @abstractmethod
    def loss_acc(self, *args, **kwargs) -> DTNS:
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
        'read_only':        False,              # sets model to be read only - wont save anything (wont even create self.nnwrap_dir)
        'do_TB':            True}               # runs TensorBard, saves in self.nnwrap_dir

    SAVE_FN_PFX = 'motorch_dna' # filename (DNA) prefix

    def __init__(
            self,
            nngraph: Optional[type(Module)]=    None,
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

    # ******************************************************************************************* NNWrap init submethods

    def _generate_name(
            self,
            given_name: Optional[str],
            timestamp: bool) -> str:
        name = f'MOTorch_{self.nngraph.__name__}' if not given_name else given_name
        if timestamp: name += f'_{stamp()}'
        return name

    # sets CPU / GPU devices for MOTorch
    def _manage_devices(self):

        self._nwwlog.debug(f'> MOTorch resolves devices, given: {self["devices"]}, torch.cuda.is_available(): {torch.cuda.is_available()}')

        self['devices'] = get_devices(
            devices=    self['devices'],
            namespace=  'torch',
            logger=     get_hi_child(self._nwwlog, 'get_devices'))
        self._nwwlog.info(f'> MOTorch will use devices: {self["devices"]}')
        # TODO: by now supported is only the first given device
        self._torch_dev = torch.device(self['devices'][0])

    # sets seed in all possible areas
    def _set_seed(self) -> None:
        # seed (https://pytorch.org/docs/stable/notes/randomness.html)
        torch.manual_seed(self['seed'])
        torch.cuda.manual_seed(self['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # builds MOTorch graph (Module)
    def _build_graph(self) -> None:

        self._nwwlog.info('MOTorch builds graph')
        #self.nngraph.__init__(self, **self._dna_nngraph)
        self._nngraph_module = self.nngraph(**self._dna_nngraph)
        self.to(self._torch_dev)
        self._nwwlog.debug(f'{self.name} (MOTorch) Module initialized!')

        try:
            self.load_ckpt()
            self._nwwlog.info(f'> MOTorch checkpoint loaded from {self.__get_ckpt_path()}')
        except Exception as e:
            self._nwwlog.info(f'> MOTorch checkpoint NOT loaded ({e})..')

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

        self.__set_training(False)
        self._nwwlog.debug(f'> set MOTorch train.mode to False..')

    def __call__(self, *args, **kwargs) -> dict:
        return torch.nn.Module.__call__(self, *args, **kwargs)

    # *********************************************************************************************** load / save / copy

    # returns path of checkpoint pickle file
    @staticmethod
    def __get_ckpt_path_static(model_dir:str, model_name:str) -> str:
        return f'{model_dir}/{model_name}.pt'

    # returns path of checkpoint pickle file
    def __get_ckpt_path(self) -> str:
        return self.__get_ckpt_path_static(self.nnwrap_dir, self.name)

    def load_ckpt(self) -> None:
        # TODO: load all that has been saved
        """
        print(f'@@@@@ {self._torch_dev}')
        INFO: for OSX (mac) self._torch_dev == 'cpu:0' and it does not load, since:
        MOTorch checkpoint NOT loaded: don't know how to restore data location of torch.storage.UntypedStorage (tagged with cpu:0)
        """
        checkpoint = torch.load(
            self.__get_ckpt_path(),
            #map_location=   'cpu') # works for for OSX
            map_location=   self._torch_dev) # INFO: to immediately place all tensors to current device (not previously saved one)
        self.load_state_dict(checkpoint['model_state_dict'])

    # saves model checkpoint
    def save_ckpt(self) -> None:
        # TODO: decide what to save
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

    # converts all values given with args & kwargs to tensors and moves to self._torch_dev (device)
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
            args = [a.to(self._torch_dev) for a in args]
            kwargs = {k: kwargs[k].to(self._torch_dev) for k in kwargs}
        return args, kwargs

    # sets self (as nn.Module) training.mode
    def __set_training(self, mode: bool):
        torch.nn.Module.train(self, mode=mode)

    # runs forward on nn.Module (with current nn.Module.training.mode - by default not training)
    # INFO: since MOTorch is a torch.nn.Module, call forward() call should be avoided, instead use just MOTorch.__call__() /self()
    def forward(
            self,
            *args,
            to_torch=                       True,   # converts given data to torch.Tensors
            to_devices=                     True,   # moves tensors to devices
            set_training: Optional[bool]=   None,   # for not None forces given training mode for torch.nn.Module
            **kwargs) -> DTNS:
        if set_training is not None: self.__set_training(set_training)
        args, kwargs = self.__torch_dev(*args, to_torch=to_torch, to_devices=to_devices, **kwargs)
        out = self._nngraph_module.forward(*args, **kwargs)
        if set_training: self.__set_training(False) # eventually roll back to default
        return out

    # runs loss calculation on nn.Module (with current nn.Module.training.mode - by default not training)
    def loss_acc(
            self,
            *args,
            to_torch=                       True,   # converts given data to torch.Tensors
            to_devices=                     True,   # moves tensors to devices
            set_training: Optional[bool]=   None,   # for not None forces given training mode for torch.nn.Module
            **kwargs) -> DTNS:
        if set_training is not None: self.__set_training(set_training)
        args, kwargs = self.__torch_dev(*args, to_torch=to_torch, to_devices=to_devices, **kwargs)
        out = self._nngraph_module.loss_acc(*args, **kwargs)
        if set_training: self.__set_training(False) # eventually roll back to default
        return out

    # runs loss calculation + update of nn.Module (by default with training.mode = True)
    def backward(
            self,
            *args,
            to_torch=           True,   # converts given data to torch.Tensors
            to_devices=         True,   # moves tensors to devices
            set_training: bool= True,
            **kwargs) -> DTNS:

        out = self.loss_acc(
            *args,
            to_torch=       to_torch,
            to_devices=     to_devices,
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

    # adds conversion to torch.Tensors and moves to device
    def load_data(
            self,
            data_TR: Dict,
            data_VL: Optional[Dict]=    None,
            data_TS: Optional[Dict]=    None):

        data = {
            'data_TR': data_TR,
            'data_VL': data_VL,
            'data_TS': data_TS}
        data_td = {}
        for k in data:
            if data[k]:
                data_td[k] = {}
                for l in data[k]:
                    d = data[k][l]
                    if type(d) is not torch.Tensor: d = torch.tensor(d)
                    d = d.to(self._torch_dev)
                    data_td[k][l] = d

        super(MOTorch, self).load_data(**data_td)

    def run_train(
            self,
            n_batches: Optional[int]=   None,
            test_freq=                  100,
            mov_avg_factor=             0.1,
            save_max=                   True,
            use_F1=                     True,
            **kwargs) -> Optional[float]:

        if not self._batcher: raise MOTorchException('MOTorch has not been given data for training, use load_data()')

        self._nwwlog.info(f'{self.name} - training starts [acc / F1 / loss]')
        self._nwwlog.info(f'data sizes (TR,VL,TS) samples: {self._batcher.get_data_size()}')

        if n_batches is None: n_batches = self['n_batches']  # take default
        self._nwwlog.info(f'batch size:             {self["batch_size"]}')
        self._nwwlog.info(f'train for num_batches:  {n_batches}')

        self.__set_training(True)

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
        if self['hpmser_mode']: ts_bIX = ts_bIX[-ten_factor:]

        while batch_IX < n_batches:

            out = self.backward(
                to_torch=   False,
                to_devices= False,
                **self._batcher.get_batch())

            acc = out['acc'] if 'acc' in out else None
            f1 = out['f1'] if 'f1' in out else None

            batch_IX += 1
            self['train_batch_IX'] += 1

            if self['do_TB']:
                self.log_TB(value=out['loss'],          tag='tr/loss',      step=self['train_batch_IX'])
                self.log_TB(value=out['gg_norm'],       tag='tr/gn',        step=self['train_batch_IX'])
                self.log_TB(value=out['gg_avt_norm'],   tag='tr/gn_avt',    step=self['train_batch_IX'])
                self.log_TB(value=out['currentLR'],     tag='tr/cLR',       step=self['train_batch_IX'])
                if acc is not None:
                    self.log_TB(value=acc,              tag='tr/acc',       step=self['train_batch_IX'])
                if f1 is not None:
                    self.log_TB(value=f1,               tag='tr/F1',       step=self['train_batch_IX'])

            if acc is not None: tr_accL.append(acc)
            if f1 is not None: tr_f1L.append(f1)
            tr_lssL.append(out['loss'])

            if batch_IX in ts_bIX:

                self.__set_training(False)
                ts_acc, ts_f1, ts_loss = self.run_test()
                self.__set_training(True)

                ts_score = ts_f1 if use_F1 else ts_acc
                if ts_score is not None:
                    ts_score_all_results.append(ts_score)
                if self['do_TB']:
                    self.log_TB(value=ts_loss,                          tag='ts/loss',              step=self['train_batch_IX'])
                    if ts_acc is not None:
                        self.log_TB(value=ts_acc,                       tag='ts/acc',               step=self['train_batch_IX'])
                    if ts_f1 is not None:
                        self.log_TB(value=ts_f1,                        tag='ts/F1',                step=self['train_batch_IX'])
                    if ts_score is not None:
                        self.log_TB(value=ts_score_mav.upd(ts_score),   tag=f'ts/{score_name}_mav', step=self['train_batch_IX'])

                tr_acc_nfo = f'{100*sum(tr_accL)/test_freq:.1f}' if acc is not None else '--'
                tr_f1_nfo =  f'{100*sum(tr_f1L)/test_freq:.1f}' if f1 is not None else '--'
                ts_acc_nfo = f'{100*ts_acc:.1f}' if ts_acc is not None else '--'
                ts_f1_nfo = f'{100*ts_f1:.1f}' if ts_f1 is not None else '--'
                self._nwwlog.info(f'# {self["train_batch_IX"]:5d} TR: {tr_acc_nfo} / {tr_f1_nfo} / {sum(tr_lssL)/test_freq:.3f} -- TS: {ts_acc_nfo} / {ts_f1_nfo} / {ts_loss:.3f}')
                tr_accL = []
                tr_f1L = []
                tr_lssL = []

                if ts_score is not None and ts_score > ts_score_max:
                    ts_score_max = ts_score
                    if not self['read_only'] and save_max: self.save_ckpt() # model is saved for max ts_score

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

            if self['do_TB']: self.log_TB(value=ts_score_wval, tag=f'ts/ts_{score_name}_wval', step=self['train_batch_IX'])

        self._nwwlog.info(f'### model {self.name} finished training')
        if ts_score_wval is not None:
            self._nwwlog.info(f' > test_{score_name}_max:  {ts_score_max:.4f}')
            self._nwwlog.info(f' > test_{score_name}_wval: {ts_score_wval:.4f}')

        self.__set_training(False)

        return ts_score_wval

    def run_test(
            self,
            set_training: Optional[bool]=   None,   # for not None sets training mode for torch.nn.Module, allows to calculate loss with training/evaluating module mode
    ) -> Tuple[Optional[float], Optional[float], float]:

        if not self._batcher: raise MOTorchException('MOTorch has not been given data for testing, use load_data() or give it while testing!')

        if set_training is not None: self.__set_training(set_training)

        batches = self._batcher.get_TS_batches()
        lossL = []
        accL = []
        f1L = []
        for batch in batches:
            out = self.loss_acc(**batch)
            lossL.append(out['loss'])
            if 'acc' in out:
                accL.append(out['acc'])
            if 'f1' in out:
                f1L.append(out['f1'])

        if set_training is not None: self.__set_training(False)  # eventually roll back to default

        acc_avg = sum(accL)/len(accL) if accL else None
        f1_avg = sum(f1L)/len(f1L) if f1L else None
        return acc_avg, f1_avg, sum(lossL)/len(lossL)

    # updates scheduler baseLR of 0 group
    def update_baseLR(self, lr: float):
        self['baseLR'] = lr # in case model will be saved >> loaded
        self._scheduler.update_base_lr0(lr)

    def __str__(self):
        s = f'MOTorch: {ParaSave.__str__(self)}\n'
        s += str(self._nngraph_module)
        return s