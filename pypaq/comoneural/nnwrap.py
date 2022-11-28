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
    - May be read only.
    - May be called (with __call__) <- runs NN FWD with given data
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
from typing import Optional, Callable, Union, Tuple, Dict

from pypaq.comoneural.batcher import Batcher
from pypaq.lipytools.little_methods import get_params, get_func_dna, prep_folder
from pypaq.lipytools.pylogger import get_pylogger, get_hi_child
from pypaq.pms.parasave import ParaSave
from pypaq.torchness.base_elements import TBwr



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
        'read_only':        False,      # sets model to be read only - wont save anything (wont even create self.nnwrap_dir)
        'do_TB':            True}       # runs TensorBard, saves in self.nnwrap_dir

    SAVE_TOPDIR = '_models'

    def __init__(
            self,
            nngraph: Optional[Union[Callable, type]]=   None,
            name: Optional[str]=                        None,
            name_timestamp=                             False,  # adds timestamp to the name
            save_topdir: Optional[str]=                 None,
            save_fn_pfx: Optional[str]=                 None,
            logger=                                     None,
            loglevel=                                   20,
            **kwargs):

        if not save_topdir: save_topdir = self.SAVE_TOPDIR
        if not save_fn_pfx: save_fn_pfx = self.SAVE_FN_PFX

        if not name and not nngraph:
            raise NNWrapException ('NNWrap ERROR: name OR nngraph must be given!')

        self.nngraph = nngraph

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
        if not _read_only: prep_folder(self.nnwrap_dir)

        if not logger:
            logger = get_pylogger(
                name=       self.name,
                add_stamp=  not name_timestamp,
                folder=     None if _read_only else self.nnwrap_dir,
                level=      loglevel)
        self._log = logger
        nng_info = self.nngraph.__name__ if self.nngraph else 'nngraph NOT GIVEN (will try to load from saved)'
        self._log.info(f'*** NNWrap *** name: {self.name} initializes for nngraph: {nng_info}')
        self._log.debug(f'> NNWrap dir: {self.nnwrap_dir}{" <- read only mode!" if _read_only else ""}')

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
            logger=                 get_hi_child(self._log, 'ParaSave'),
            **self._dna)
        self.check_params_sim(params= list(self.SPEC_KEYS) + list(self.INIT_DEFAULTS.keys())) # safety check

        self._manage_devices()

        self._set_seed()

        self._build_graph()

        self._TBwr = TBwr(logdir=self.nnwrap_dir)  # TensorBoard writer

        self._batcher = None

        self._log.debug(str(self))
        self._log.info(f'NNWrap init finished!')

    # ******************************************************************************************* NNWrap init submethods

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
            self._log.error(msg)
            raise NNWrapException(msg)

        # get defaults of given nngraph (object.__init__ or callable)
        nngraph_func = self.nngraph.__init__ if type(self.nngraph) is object else self.nngraph
        nngraph_func_params = get_params(nngraph_func)
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

        dna_with_logger = {}
        dna_with_logger.update(self._dna)
        dna_with_logger['logger'] = get_hi_child(
            logger= self._log,
            name=   f'{self.name}_sublogger'),
        self._dna_nngraph = get_func_dna(nngraph_func, dna_with_logger)

        not_used_kwargs = {}
        for k in kwargs:
            if k not in self._dna_nngraph:
                not_used_kwargs[k] = kwargs[k]

        self._log.debug(f'> {self.name} DNA sources:')
        self._log.debug(f'>> class INIT_DEFAULTS:        {self.INIT_DEFAULTS}')
        self._log.debug(f'>> nngraph defaults:           {nngraph_func_params_defaults}')
        self._log.debug(f'>> DNA saved:                  {dna_saved}')
        self._log.debug(f'>> given kwargs:               {kwargs}')
        self._log.debug(f'> resolved DNA:')
        self._log.debug(f'nngraph complete DNA:          {self._dna_nngraph}')
        self._log.debug(f'>> kwargs not used by nngraph: {not_used_kwargs}')
        self._log.debug(f'{self.name} complete DNA:      {self._dna}')

    # manages NNWrap CUDA / CPU devices
    @abstractmethod
    def _manage_devices(self) -> None: pass

    # sets NNWrap seed in all possible areas
    @abstractmethod
    def _set_seed(self) -> None: pass

    # builds NNWrap graph
    @abstractmethod
    def _build_graph(self) -> None: pass

    @abstractmethod
    def __call__(self, *args, **kwargs): pass

    # *********************************************************************************************** load / save / copy

    # (re)loads model checkpoint
    @abstractmethod
    def load_ckpt(self) -> None: pass

    # saves model checkpoint
    @abstractmethod
    def save_ckpt(self) -> None: pass

    # saves NNWrap (ParaSave DNA and model checkpoint)
    def save(self) -> None:
        if self['read_only']: raise NNWrapException('read only NNWrap cannot be saved!')
        self.save_dna()
        self.save_ckpt()
        self._log.info(f'NNWrap {self.name} saved')

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
            noise: float=                       0.03) -> None: pass

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
    def load_data(self, data: Dict):

        if 'train' not in data:
            msg = 'given data should be a dict with at least "train" key present!'
            self._log.error(msg)
            raise NNWrapException(msg)

        self._batcher = Batcher(
            data_TR=        data['train'],
            data_VL=        data['valid'] if 'valid' in data else None,
            data_TS=        data['test'] if 'test' in data else None,
            batch_size=     self['batch_size'],
            batching_type=  'random_cov',
            logger=         get_hi_child(self._log, 'Batcher'))

     # trains model, should save saves max test scored model, returns test score
    @abstractmethod
    def train(
            self,
            data=                       None,
            n_batches: Optional[int]=   None,
            test_freq=                  100,    # number of batches between tests, model SHOULD BE tested while training
            mov_avg_factor=             0.1,
            save=                       True,   # allows to save model while training (after max test)
            **kwargs) -> float: pass

    # tests model, returns accuracy and loss (average)
    @abstractmethod
    def test(self, data=None, **kwargs) -> Tuple[float,float]: pass

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
        else: self._log.warning(f'NNWrap {self.name} cannot log TensorBoard since do_TB flag is False!')

    # returns nice string about self
    @abstractmethod
    def __str__(self) -> str: pass