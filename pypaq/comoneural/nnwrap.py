from abc import ABC, abstractmethod
from typing import Optional, Callable, Union

from pypaq.lipytools.little_methods import get_params, get_func_dna, prep_folder
from pypaq.lipytools.pylogger import get_pylogger, get_hi_child
from pypaq.pms.parasave import ParaSave
from pypaq.torchness.base_elements import TBwr



class NNWrapException(Exception):
    pass


class NNWrap(ParaSave, ABC):

    # TODO: extend defaults

    # restricted keys for fwd_func DNA and return DNA (if they appear in kwargs, should be named exactly like below)
    _SPEC_KEYS: set = {
        'loss',                         # loss
        'acc',                          # accuracy
        'f1'}                           # F1

    # defaults (state), may be overridden with kwargs or nngraph (__init__?) attributes
    _INIT_DEFAULTS: dict = {
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
        'read_only':        False,      # sets model to be read only - wont save anything (wont even create self.model_dir)
        'do_TB':            True}       # runs TensorBard, saves in self.model_dir

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

        self.model_dir = f'{save_topdir}/{self.name}'
        if not _read_only: prep_folder(self.model_dir)

        if not logger:
            logger = get_pylogger(
                name=       self.name,
                add_stamp=  not name_timestamp,
                folder=     None if _read_only else self.model_dir,
                level=      loglevel)
        self._log = logger
        nng_info = self.nngraph.__name__ if self.nngraph else 'nngraph NOT GIVEN (will try to load from saved)'
        self._log.info(f'*** NNWrap *** name: {self.name} initializes for nngraph: {nng_info}')
        self._log.debug(f'> NNWrap dir: {self.model_dir}{" <- read only mode!" if _read_only else ""}')

        self._dna = {
            'nngraph':      self.nngraph,
            'model_dir':    self.model_dir}
        self._manage_dna(
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx,
            **kwargs)

        ParaSave.__init__(
            self,
            lock_managed_params=    True,
            logger=                 get_hi_child(self._log, 'ParaSave'),
            **self._dna)
        self.check_params_sim(params= list(self._SPEC_KEYS) + list(self._INIT_DEFAULTS.keys())) # safety check

        self._manage_devices()

        self._set_seed()

        self._build_graph()

        self.__TBwr = TBwr(logdir=self.model_dir)  # TensorBoard writer

        self._batcher = None

        self._log.debug(str(self))
        self._log.info(f'NNWrap init finished!')

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

        if not self.nngraph and 'nngraph' not in dna_saved:
            msg = 'nngraph was not given and has not been found in saved, cannot continue!'
            self._log.error(msg)
            raise NNWrapException(msg)

        # get defaults of given nngraph (object.__init__ or callable)
        _nngraph_func = self.nngraph.__init__ if type(self.nngraph) is object else self.nngraph
        _nngraph_func_params = get_params(_nngraph_func)
        _nngraph_func_params_defaults = _nngraph_func_params['with_defaults']   # get init params defaults
        if 'logger' in _nngraph_func_params_defaults: _nngraph_func_params_defaults.pop('logger')

        # update in proper order
        self._dna.update(self._INIT_DEFAULTS)
        self._dna.update(_nngraph_func_params_defaults)
        self._dna.update(dna_saved)
        self._dna.update(kwargs)          # update with kwargs given NOW by user
        self._dna.update({
            'name':         self.name,
            'save_topdir':  save_topdir,
            'save_fn_pfx':  save_fn_pfx})

        dna_with_logger = {}
        dna_with_logger.update(self._dna)
        dna_with_logger['logger'] = get_hi_child(
            logger= self._log,
            name=   f'{self.name}_sublogger'),
        self._dna_nngraph = get_func_dna(_nngraph_func, dna_with_logger)

        not_used_kwargs = {}
        for k in kwargs:
            if k not in self._dna_nngraph:
                not_used_kwargs[k] = kwargs[k]

        self._log.debug(f'> {self.name} DNA sources:')
        self._log.debug(f'>> {self.name} _INIT_DEFAULTS: {self._INIT_DEFAULTS}')
        self._log.debug(f'>> nngraph defaults:           {_nngraph_func_params_defaults}')
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



