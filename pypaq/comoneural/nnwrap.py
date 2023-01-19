"""
 2022 (c) piteren

 NNWrap is an abstract interface of object that wraps neural network (NN) and adds some features.

    NNWrap:
    - Builds FWD and OPT graph. FWD graph is built with nngraph which may be
      an object or callable. FWD graph should be built from inputs to loss.
    - Manages one NNWrap folder (subfolder of SAVE_TOPDIR named with NNWrap name)
      for all NNWrap data (logs, params, checkpoints). NNWrap supports
      serialization into this folder.
    - Extends ParaSave, manages all init parameters. Properly resolves parameters
      using all possible sources, saves and loads them from NNWrap folder.
    - Parameters are kept in self as a Subscriptable to be easily accessed.
    - Properly resolves and holds name of object, adds stamp if needed.
    - Supports / creates logger.
    - May be read only (prevents save over).
    - May be called (with __call__) <- runs NN FWD with given data
    - May be called BWD with backward() <- runs gradient backprop for given data
    - Supports hpmser mode.
    - Manages seed and guarantees reproducibility.
    - Manages GPU / CPU devices used by NN.
    - Adds TensorBoard support.
    - Defines / implements save / load / copy of whole NNWrap (ParaSave + NN checkpoint).
    - Defines interface of baseline training & testing with data loaded to Batcher.
    - Defines / implements GX.
    - Adds some sanity checks.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Callable, Union, Tuple, Dict

from pypaq.comoneural.batcher import Batcher, split_data_TR
from pypaq.lipytools.little_methods import get_params, get_func_dna
from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.pylogger import get_pylogger, get_hi_child
from pypaq.pms.parasave import ParaSave
from pypaq.torchness.tbwr import TBwr


class NNWrapException(Exception):
    pass


class NNWrap(ParaSave, ABC):

    # restricted keys for fwd_func DNA and return DNA (if they appear in kwargs, should be named exactly like below)
    SPEC_KEYS: set = {
        'loss',                         # loss
        'acc',                          # accuracy
        'f1'}                           # F1

    # defaults (state), may be overridden with kwargs or nngraph (__init__?) attributes
    INIT_DEFAULTS: dict = {
        'seed':             123,        # seed
        'devices':          -1,         # :DevicesParam (check pypaq.mpython.devices)
            # training
        'batch_size':       64,         # training batch size
        'n_batches':        1000,       # default length of training
        'train_batch_IX':   0,          # default (starting) batch index (counter)
            # LR management
        'baseLR':           3e-4,
        'warm_up':          None,
        'ann_base':         None,
        'ann_step':         1.0,
        'n_wup_off':        2.0,
            # gradients clipping parameters
        'clip_value':       None,
        'avt_SVal':         0.1,
        'avt_window':       100,
        'avt_max_upd':      1.5,
        'do_clip':          False,
            # other
        'hpmser_mode':      False,      # it will set model to be read_only and quiet when running with hpmser
        'read_only':        False,      # sets model to be read only - won't save anything (won't even create self.nnwrap_dir)
        'do_TB':            True}       # runs TensorBard, saves in self.nnwrap_dir

    SAVE_TOPDIR = '_models'

    def __init__(
            self,
            nngraph: Optional[Union[Callable, type]]=   None,   # function (Callable) or type (of object) that defines NN
            name: Optional[str]=                        None,
            name_timestamp=                             False,  # adds timestamp to the name
            save_topdir: Optional[str]=                 None,
            save_fn_pfx: Optional[str]=                 None,
            logger=                                     None,
            loglevel=                                   20,
            **kwargs):

        if not name and not nngraph:
            raise NNWrapException('NNWrap ERROR: name OR nngraph must be given!')

        if 'device' in kwargs:
            raise NNWrapException('NNWrap uses \'devices\' param to set devices, not \'device\'!')

        if not save_topdir: save_topdir = self.SAVE_TOPDIR
        if not save_fn_pfx: save_fn_pfx = self.SAVE_FN_PFX

        self.nngraph = nngraph # INFO: here we save TYPE

        self.name = self._generate_name(
            given_name= name,
            timestamp=  name_timestamp)

        # some early overrides

        if kwargs.get('hpmser_mode', False):
            loglevel = 50
            kwargs['read_only'] = True

        if kwargs.get('read_only', False):
            kwargs['do_TB'] = False

        _read_only = kwargs.get('read_only', False)

        self.nnwrap_dir = f'{save_topdir}/{self.name}'

        if not logger:
            logger = get_pylogger(
                name=       self.name,
                add_stamp=  False,
                folder=     None if _read_only else self.nnwrap_dir,
                level=      loglevel)
        self._nwwlog = logger
        nng_info = self.nngraph.__name__ if self.nngraph else 'nngraph NOT GIVEN (will try to load from saved)'
        self._nwwlog.info(f'*** NNWrap *** name: {self.name} initializes for nngraph: {nng_info}')
        self._nwwlog.info(f'> NNWrap dir: {self.nnwrap_dir}{" <- read only mode!" if _read_only else ""}')

        self._dna = dict(
            nngraph=    self.nngraph,
            model_dir=  self.nnwrap_dir)
        self._manage_dna(
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx,
            **kwargs)

        ParaSave.__init__(
            self,
            lock_managed_params=    True,
            logger=                 get_hi_child(self._nwwlog),
            **self._dna)

        # params names safety check
        pms = sorted(list(self.SPEC_KEYS) + list(self.INIT_DEFAULTS.keys()) + list(kwargs.keys()))
        found = self.check_params_sim(params=pms)
        if found:
            self._nwwlog.warning('NNWrap was asked to check for params similarity and found:')
            for pa, pb in found: self._nwwlog.warning(f'> params \'{pa}\' and \'{pb}\' are too CLOSE !!!')

        self._manage_devices()

        self._set_seed()

        self._build_graph()

        self._TBwr = TBwr(logdir=self.nnwrap_dir)  # TensorBoard writer

        self._batcher = None

        self._nwwlog.debug(str(self))
        self._nwwlog.info(f'NNWrap init finished!')

    # ************************************************************************************************** init submethods

    # generates NNWrap name
    @abstractmethod
    def _generate_name(
            self,
            given_name: Optional[str],
            timestamp: bool) -> str:
        pass

    # manages dna, reports
    def _manage_dna(
            self,
            save_topdir: str,
            save_fn_pfx: str,
            **kwargs) -> None:

        # load dna from folder
        dna_saved = self.load_dna(
            name=           self.name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx)

        # in case 'nngraph' was not given with init, try to get it from saved
        if not self.nngraph:
            self.nngraph = dna_saved.get('nngraph', None)

        if not self.nngraph:
            msg = 'nngraph was not given and has not been found in saved, cannot continue!'
            self._nwwlog.error(msg)
            raise NNWrapException(msg)

        # get defaults of given nngraph (object.__init__ or callable)
        nngraph_func = self.nngraph.__init__ if type(self.nngraph) is object else self.nngraph
        nngraph_func_params = get_params(nngraph_func)
        for pn in ['device','devices']:
            if pn in nngraph_func_params['without_defaults'] or pn in nngraph_func_params['with_defaults']:
                self._nwwlog.warning(f'NNWrap nngraph \'{pn}\' parameter wont be used, since devices are managed by NNWrap')
        nngraph_func_params_defaults = nngraph_func_params['with_defaults']   # get init params defaults
        if 'logger' in nngraph_func_params_defaults: nngraph_func_params_defaults.pop('logger')

        # update in proper order
        self._dna.update(self.INIT_DEFAULTS)
        self._dna.update(nngraph_func_params_defaults)
        self._dna.update(dna_saved)
        self._dna.update(kwargs)  # update with kwargs given NOW by user
        self._dna.update({
            'name':         self.name,
            'save_topdir':  save_topdir,
            'save_fn_pfx':  save_fn_pfx})

        dna_with_nwwlog = {}
        dna_with_nwwlog.update(self._dna)
        dna_with_nwwlog['logger'] = self._nwwlog
        self._dna_nngraph = get_func_dna(nngraph_func, dna_with_nwwlog)

        not_used_kwargs = {}
        for k in kwargs:
            if k not in self._dna_nngraph:
                not_used_kwargs[k] = kwargs[k]

        self._nwwlog.debug(f'> {self.name} DNA sources:')
        self._nwwlog.debug(f'>> class INIT_DEFAULTS:        {self.INIT_DEFAULTS}')
        self._nwwlog.debug(f'>> nngraph defaults:           {nngraph_func_params_defaults}')
        self._nwwlog.debug(f'>> DNA saved:                  {dna_saved}')
        self._nwwlog.debug(f'>> given kwargs:               {kwargs}')
        self._nwwlog.debug(f'> resolved DNA:')
        self._nwwlog.debug(f'nngraph complete DNA:          {self._dna_nngraph}')
        self._nwwlog.debug(f'>> kwargs not used by nngraph: {not_used_kwargs}')
        self._nwwlog.debug(f'{self.name} complete DNA:      {self._dna}')

    # manages NNWrap CUDA / CPU devices
    @abstractmethod
    def _manage_devices(self) -> None: pass

    # sets NNWrap seed in all possible areas
    @abstractmethod
    def _set_seed(self) -> None: pass

    # builds NNWrap graph
    @abstractmethod
    def _build_graph(self) -> None: pass

    # **************************************************************************** model call (run NN with data) methods

    @abstractmethod
    def __call__(self, *args, **kwargs) -> dict: pass

    @abstractmethod
    def loss(self, *args, **kwargs) -> dict: pass

    @abstractmethod
    def backward(self, *args, **kwargs) -> dict: pass

    # *********************************************************************************************** load / save / copy

    # (re)loads model checkpoint
    @abstractmethod
    def load_ckpt(self) -> None: pass

    # saves model checkpoint
    @abstractmethod
    def save_ckpt(self) -> None: pass

    # saves NNWrap (ParaSave DNA and model checkpoint)
    def save(self):
        if self['read_only']: raise NNWrapException('read only NNWrap cannot be saved!')
        self.save_dna()
        self.save_ckpt()
        self._nwwlog.info(f'NNWrap {self.name} saved')

    # copies just model checkpoint
    @classmethod
    @abstractmethod
    def copy_checkpoint(
            cls,
            name_src: str,
            name_trg: str,
            save_topdir_src: Optional[str]= None,
            save_topdir_trg: Optional[str]= None) -> None: pass

    # copies full NNWrap folder (DNA & checkpoints)
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

        cls.copy_saved_dna(
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

    # GX for two NNWrap checkpoints
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
            noise: float=                       0.03) -> None:
        raise NotImplementedError

    # performs GX on saved NNWrap objects, without even building child objects
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

        cls.gx_saved_dna(
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

    # loads data to Batcher
    def load_data(
            self,
            data: Dict[str, np.ndarray],
            split_VL: float=    0.0,
            split_TS: float=    0.0):

        data_TR, data_VL, data_TS = split_data_TR(
            data=       data,
            split_VL=   split_VL,
            split_TS=   split_TS,
            seed=       self['seed'])

        self._batcher = Batcher(
            data_TR=        data_TR,
            data_VL=        data_VL,
            data_TS=        data_TS,
            batch_size=     self['batch_size'],
            batching_type=  'random_cov',
            logger=         get_hi_child(self._nwwlog, 'Batcher'))

     # trains model, returns optional test score
    def run_train(
            self,
            data: Optional[Dict[str, np.ndarray]]=  None,
            split_VL: float=                        0.0,
            split_TS: float=                        0.0,
            n_batches: Optional[int]=               None,
            test_freq=                              100,    # number of batches between tests, model SHOULD BE tested while training
            mov_avg_factor=                         0.1,
            save_max=                               True,   # allows to save model while training (after max test)
            use_F1=                                 True,   # uses F1 as a train/test score (not acc)
            **kwargs) -> Optional[float]:

        if data:
            self.load_data(
                data=       data,
                split_VL=   split_VL,
                split_TS=   split_TS)

        if not self._batcher: raise NNWrapException(f'{self.name} has not been given data for training, use load_data()')

        self._nwwlog.info(f'{self.name} - training starts [acc / F1 / loss]')
        self._nwwlog.info(f'data sizes (TR,VL,TS) samples: {self._batcher.get_data_size()}')

        if n_batches is None: n_batches = self['n_batches']  # take default
        self._nwwlog.info(f'batch size:             {self["batch_size"]}')
        self._nwwlog.info(f'train for num_batches:  {n_batches}')

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

            out = self.backward(**self._batcher.get_batch())

            loss = out['loss']
            acc = out['acc'] if 'acc' in out else None
            f1 = out['f1'] if 'f1' in out else None

            batch_IX += 1
            self['train_batch_IX'] += 1

            if self['do_TB']:
                self.log_TB(value=loss,                 tag='tr/loss',      step=self['train_batch_IX'])
                self.log_TB(value=out['gg_norm'],       tag='tr/gn',        step=self['train_batch_IX'])
                self.log_TB(value=out['gg_avt_norm'],   tag='tr/gn_avt',    step=self['train_batch_IX'])
                self.log_TB(value=out['currentLR'],     tag='tr/cLR',       step=self['train_batch_IX'])
                if acc is not None:
                    self.log_TB(value=acc,              tag='tr/acc',       step=self['train_batch_IX'])
                if f1 is not None:
                    self.log_TB(value=f1,               tag='tr/F1',       step=self['train_batch_IX'])

            if acc is not None: tr_accL.append(acc)
            if f1 is not None: tr_f1L.append(f1)
            tr_lssL.append(loss)

            if batch_IX in ts_bIX:

                ts_loss, ts_acc, ts_f1 = self.run_test()

                ts_score = ts_f1 if use_F1 else ts_acc
                if ts_score is not None:
                    ts_score_all_results.append(ts_score)
                if self['do_TB']:
                    if ts_loss is not None:
                        self.log_TB(value=ts_loss,                      tag='ts/loss',              step=self['train_batch_IX'])
                    if ts_acc is not None:
                        self.log_TB(value=ts_acc,                       tag='ts/acc',               step=self['train_batch_IX'])
                    if ts_f1 is not None:
                        self.log_TB(value=ts_f1,                        tag='ts/F1',                step=self['train_batch_IX'])
                    if ts_score is not None:
                        self.log_TB(value=ts_score_mav.upd(ts_score),   tag=f'ts/{score_name}_mav', step=self['train_batch_IX'])

                tr_acc_nfo = f'{100*sum(tr_accL)/test_freq:.1f}' if acc is not None else '--'
                tr_f1_nfo =  f'{100*sum(tr_f1L)/test_freq:.1f}' if f1 is not None else '--'
                tr_loss_nfo = f'{sum(tr_lssL)/test_freq:.3f}'
                ts_acc_nfo = f'{100*ts_acc:.1f}' if ts_acc is not None else '--'
                ts_f1_nfo = f'{100*ts_f1:.1f}' if ts_f1 is not None else '--'
                ts_loss_nfo = f'{ts_loss:.3f}' if ts_loss is not None else '--'
                self._nwwlog.info(f'# {self["train_batch_IX"]:5d} TR: {tr_acc_nfo} / {tr_f1_nfo} / {tr_loss_nfo} -- TS: {ts_acc_nfo} / {ts_f1_nfo} / {ts_loss_nfo}')
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

        return ts_score_wval

    # tests model, returns: optional loss (average), optional accuracy, optional F1
    # optional loss <- since there may be not TS batches
    def run_test(
            self,
            data: Optional[Dict[str, np.ndarray]]=  None,
            split_VL: float=                        0.0,
            split_TS: float=                        1.0, # if data for test will be given above, by default MOTorch will be tested on ALL
            **kwargs) -> Tuple[Optional[float], Optional[float], Optional[float]]:

        if data:
            self.load_data(
                data=       data,
                split_VL=   split_VL,
                split_TS=   split_TS)

        if not self._batcher: raise NNWrapException(f'{self.name} has not been given data for testing, use load_data() or give it while testing!')

        batches = self._batcher.get_TS_batches()
        lossL = []
        accL = []
        f1L = []
        for batch in batches:
            out = self.loss(**batch)
            lossL.append(out['loss'])
            if 'acc' in out: accL.append(out['acc'])
            if 'f1' in out:  f1L.append(out['f1'])

        acc_avg = sum(accL)/len(accL) if accL else None
        f1_avg = sum(f1L)/len(f1L) if f1L else None
        loss_avg = sum(lossL)/len(lossL) if lossL else None
        return loss_avg, acc_avg, f1_avg

    # *********************************************************************************************** other / properties

    # updates model baseLR
    @abstractmethod
    def update_baseLR(self, lr: float) -> None: pass

    @property
    def tbwr(self):
        return self._TBwr

    # logs value to TB
    def log_TB(
            self,
            value,
            tag: str,
            step: int) -> None:
        if self['do_TB']: self._TBwr.add(value=value, tag=tag, step=step)
        else: self._nwwlog.warning(f'NNWrap {self.name} cannot log TensorBoard since do_TB flag is False!')

    @property
    def logger(self):
        return self._nwwlog

    @property
    def size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_devices(self): pass

    # returns nice string about self
    @abstractmethod
    def __str__(self) -> str: pass