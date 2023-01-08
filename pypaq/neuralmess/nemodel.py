"""
 2019 (c) piteren

 NEModel implements NNWrap interface with TensorFlow.

    nngraph for NEModel:
        - should build complete model forward graph (FWD): from PH (placeholders) to loss tensor
        - function should return a dict with: PH, tensors, variables lists
            - dict keys should meet naming policy
                - list of special keys to consider while building fwd_func is under SPEC_KEYS
                - OPT part will be build only if there is a 'loss' tensor returned
                - dict may contain variables_lists and single variable under keys with 'var' in name
                    - variables returned under 'train_vars' key are optimized (if 'train_vars' key is not present all trainable vars are optimized)
                    - sub-lists of variables will serve for separate savers (saved in subfolders)
    opt_func:
        - rather should not be replaced, but if so then it should accept train_vars and gradients parameters
        - should return 'optimizer'

 NEModel class extends NNWrap with:
    - GPU automated management with multi-GPU training on towers (places model graph elements across available devices)
    - builds optimization (OPT) graph with default OPT function
        - calculates gradients for every tower >> averages them
        - AVT gradient clipping and scaled LR (warmup, annealing)
    - exemplary FWD graph function
"""

import numpy as np
import os
from typing import Optional, Callable, Tuple, List
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pypaq.lipytools.little_methods import short_scin, stamp, get_params, get_func_dna
from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.pylogger import get_pylogger, get_hi_child
from pypaq.mpython.devices import mask_cuda, get_devices
from pypaq.pms.parasave import ParaSave
from pypaq.comoneural.nnwrap import NNWrap, NNWrapException
from pypaq.neuralmess.get_tf import tf
from pypaq.neuralmess.base_elements import num_var_floats, lr_scaler, gc_loss_reductor, log_vars, mrg_ckpts
from pypaq.neuralmess.layers import lay_dense
from pypaq.neuralmess.multi_saver import MultiSaver


# default FWD function (forward graph), it is given as an exemplary implementation
def fwd_graph(
        name: str,
        seq_len: int,
        emb_num: int=   50,
        emb_width: int= 100,
        n_labels: int=  2,
        seed: int=      321,
        baseLR=         0.003):
    with tf.variable_scope(name):

        in_PH = tf.placeholder(
            name=           'in_PH',
            dtype=          tf.int32,
            shape=          [None, seq_len])
        labels_PH = tf.placeholder(
            name=           'labels_PH',
            dtype=          tf.int32,
            shape=          [None, 1])
        emb = tf.get_variable( # some embeddings
            name=       'emb',
            shape=      [emb_num, emb_width],
            dtype=      tf.float32)

        feats = tf.nn.embedding_lookup(params=emb, ids=in_PH)

        logits = lay_dense(
            input=      feats,
            units=      n_labels,
            name=       'logits',
            seed=       seed)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels= labels_PH,
            logits= logits)

        pred = tf.argmax(logits, axis=-1)

        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(pred, dtype=tf.int32), labels_PH), dtype=tf.float32))

    return {
        'in_PH':    in_PH,
        'emb':      emb,
        'logits':   logits,
        'loss':     loss,
        'acc':      acc}

# default OPT function (optimization graph), should be used in most scenarios
def opt_graph(
        train_vars,
        gradients,
        opt_class=          tf.train.AdamOptimizer,
            # LR management (check pypaq.neuralmess.base_elements.lr_scaler)
        baseLR=             3e-4,
        warm_up=            None,
        ann_base=           None,
        ann_step=           1.0,
        n_wup_off: float=   2.0,
            # gradients clipping parameters (check pypaq.neuralmess.base_elements.gc_loss_reductor)
        clip_value=         None,
        avt_SVal=           0.1,
        avt_window=         100,
        avt_max_upd=        1.5,
        do_clip=            False,
        logger=             None):

    if not logger: logger = get_pylogger()

    g_step = tf.get_variable(  # global step variable
        name=           'g_step',
        shape=          [],
        trainable=      False,
        initializer=    tf.constant_initializer(0),
        dtype=          tf.int32)

    baseLR_var = tf.get_variable(  # base LR variable
        name=           'baseLR',
        shape=          [],
        trainable=      False,
        initializer=    tf.constant_initializer(baseLR),
        dtype=          tf.float32)

    scaled_LR = lr_scaler(
        baseLR=         baseLR_var,
        g_step=         g_step,
        warm_up=        warm_up,
        ann_base=       ann_base,
        ann_step=       ann_step,
        n_wup_off=      n_wup_off,
        logger=         logger)['scaled_LR']

    # updates with: optimizer, gg_norm, gg_avt_norm
    loss_reductorD = gc_loss_reductor(
        optimizer=      opt_class(learning_rate=scaled_LR),
        vars=           train_vars,
        g_step=         g_step,
        gradients=      gradients,
        clip_value=     clip_value,
        avt_SVal=       avt_SVal,
        avt_window=     avt_window,
        avt_max_upd=    avt_max_upd,
        do_clip=        do_clip,
        logger=         logger)

    # select OPT vars
    opt_vars = tf.global_variables(scope=tf.get_variable_scope().name)

    logger.debug(f' ### opt_vars: {len(opt_vars)} floats: {short_scin(num_var_floats(opt_vars))} ({opt_vars[0].device})')
    logger.log(5, log_vars(opt_vars))

    rd = {}
    rd.update({
        'g_step':       g_step,
        'baseLR_var':   baseLR_var,
        'scaled_LR':    scaled_LR,
        'opt_vars':     opt_vars})
    rd.update(loss_reductorD)
    return rd


class NEModelException(NNWrapException):
    pass

# NEModel Base class, implements most features (but not saving)
class NEModel(NNWrap):

    SPEC_KEYS = {
        'train_vars',   # list of variables to train (may be returned, otherwise all trainable are taken)
        'opt_vars',     # list of variables returned by opt_func
        'loss',         # loss
        'acc',          # accuracy
        'f1'}           # F1

    # some defaults of NEModel are in default opt_func (opt_graph) above
    INIT_DEFAULTS = {
        'seed':                 123,        # seed for TF and numpy
        'devices':              -1,         # : DevicesParam (check pypaq.mpython.devices)
            # training
        'batch_size':           64,         # training batch size
        'n_batches':            1000,       # default length of training
        'train_batch_IX':       0,          # default (starting) batch index (counter)
            # other
        'hpmser_mode':          False,      # it will set model to be read_only and quiet when running with hpmser
        'savers_names':         (None,),    # names of savers for MultiSaver
        'load_saver':           True,       # Optional[bool or str] for None/False does not load, for True loads default
        'read_only':            False,      # sets model to be read only - wont save anything (wont even create self.nnwrap_dir)
        'do_TB':                True,       # runs TensorBard, saves in self.nnwrap_dir
        'silent_TF_warnings':   False,      # turns off TF warnings
        'sep_device':           True,       # separate first device for variables, gradients_avg, optimizer (otherwise those ar placed on the first FWD calculations tower)
        'collocate_GWO':        False}      # collocates gradient calculations with tf.OPs (gradients are calculated on every tower with its operations, but remember that vars are on one device...) (otherwise with first FWD calculations tower)

    SAVE_FN_PFX = 'nemodel_dna' # filename (DNA) prefix

    def __init__(
            self,
            nngraph: Optional[Callable]=    None,       # forward function
            opt_func: Optional[Callable]=   opt_graph,  # optimization function building optimization (OPT) graph (from train_vars & gradients to optimizer)
            **kwargs):

        if kwargs.get('silent_TF_warnings', False):
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            warnings.filterwarnings('ignore')

        self._dna_opt_func = {}
        self._gFWD = [] # list of dicts of all FWD graphs (from all devices)
        self._graph = None
        self._session = None
        self._saver = None

        NNWrap.__init__(
            self,
            nngraph=    nngraph,
            opt_func=   opt_func, # give with kwargs
            **kwargs)

    # ******************************************************************************************* NNWrap init submethods

    def _generate_name(
            self,
            given_name: Optional[str],
            timestamp: bool) -> str:
        name = f'NEModel_{self.nngraph.__name__}' if not given_name else given_name
        if timestamp: name += f'_{stamp()}'
        return name

    # overrides NNWrap version a little, since NEModel has TWO graph functions
    def _manage_dna(
            self,
            save_topdir: str,
            save_fn_pfx: str,
            **kwargs) -> None:

        # load dna from folder
        dna_saved = NEModel.load_dna(
            name=           self.name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx)

        # in case 'nngraph' was not given with init, try to get it from saved
        if not self.nngraph:
            self.nngraph = dna_saved.get('nngraph', None)

        if not self.nngraph:
            msg = 'nngraph was not given and has not been found in saved, cannot continue!'
            self._nwwlog.error(msg)
            raise NEModelException(msg)

        # get defaults of given nngraph and opt_func
        nngraph_func_params = get_params(self.nngraph)
        nngraph_func_params_defaults = nngraph_func_params['with_defaults']  # get nngraph defaults
        opt_func = kwargs['opt_func']
        opt_func_params_defaults = get_params(opt_func)['with_defaults'] if opt_func else {}
        if 'logger' in nngraph_func_params_defaults: nngraph_func_params_defaults.pop('logger')
        if 'logger' in opt_func_params_defaults: opt_func_params_defaults.pop('logger')

        self._dna.update(self.INIT_DEFAULTS)
        self._dna.update(opt_func_params_defaults)
        self._dna.update(nngraph_func_params_defaults)
        self._dna.update(dna_saved)
        self._dna.update(kwargs)  # update with kwargs (params of FWD & OPT) given NOW by user
        self._dna.update({
            'name':         self.name,
            'save_topdir':  save_topdir,
            'save_fn_pfx':  save_fn_pfx})

        dna_with_nwwlog = {}
        dna_with_nwwlog.update(self._dna)
        dna_with_nwwlog['logger'] = get_hi_child(
            logger= self._nwwlog,
            name=   f'{self.name}_sublogger')
        self._dna_nngraph = get_func_dna(self.nngraph, dna_with_nwwlog)
        self._dna_opt_func = get_func_dna(opt_func, dna_with_nwwlog)

        not_used_kwargs = {}
        for k in kwargs:
            if k not in self._dna_nngraph and k not in self._dna_opt_func:
                not_used_kwargs[k] = kwargs[k]

        self._nwwlog.debug(f'> {self.name} DNA sources:')
        self._nwwlog.debug(f'>> class INIT_DEFAULTS:        {self.INIT_DEFAULTS}')
        self._nwwlog.debug(f'>> nngraph defaults:           {nngraph_func_params_defaults}')
        self._nwwlog.debug(f'>> opt_func defaults:          {opt_func_params_defaults}')
        self._nwwlog.debug(f'>> DNA saved:                  {dna_saved}')
        self._nwwlog.debug(f'>> given kwargs:               {kwargs}')
        self._nwwlog.debug(f'> resolved DNA:')
        self._nwwlog.debug(f'nngraph complete DNA:          {self._dna_nngraph}')
        self._nwwlog.debug(f'opt_func complete DNA:         {self._dna_opt_func}')
        self._nwwlog.debug(f'>> kwargs not used by graphs : {not_used_kwargs}')
        self._nwwlog.debug(f'{self.name} complete DNA:      {self._dna}')

    def _manage_devices(self):

        self['devices'] = get_devices(
            devices=    self['devices'],
            namespace=  'TF1',
            logger=     get_hi_child(self._nwwlog, 'get_devices'))

        devices_other = []
        devices_gpu = []
        for device in self['devices']:
            if 'GPU' in device: devices_gpu.append(device)
            else: devices_other.append(device)

        # prepare ids & rewrite GPU devices
        ids = []
        if devices_gpu:
            devices_gpu = [f'/device:GPU:{ix}' for ix in range(len(devices_gpu))]
            ids = [dev[12:] for dev in devices_gpu]

        self['devices'] = devices_other + devices_gpu

        self._nwwlog.debug(f' > masking GPU devices: {ids}')
        mask_cuda(ids) # mask GPU devices

        # report devices
        if len(self['devices'])==1:
            if 'CPU' in self['devices'][0]: self._nwwlog.info(f'NEModel builds CPU device setup')
            else:                           self._nwwlog.info(f'NEModel builds single-GPU setup: {self["devices"]}')
        else:                               self._nwwlog.info(f'NEModel builds multi-dev setup for {len(self["devices"])} devices')

        if len(self['devices'])<3: self['sep_device'] = False # SEP is available for 3 or more devices

    # sets NNWrap seed in all possible areas
    def _set_seed(self) -> None:
        tf.set_random_seed(self['seed'])
        np.random.seed(self['seed'])

    # builds graph (FWD & OPT) and manages surroundings
    def _build_graph(self) -> None:

        # build FWD graph(s) >> manage variables >> build OPT graph
        self._graph = tf.Graph()
        with self._graph.as_default():

            self._nwwlog.debug(f'NEModel set TF & NP seed to {self["seed"]}')

            # builds graph @SEP, this graph wont be run, it is only needed to place variables, if not vars_sep >> variables will be placed with first tower
            if self['sep_device']:
                self._nwwlog.debug(f'NEModel places VARs on {self["devices"][0]}...')
                with tf.device(self['devices'][0]):
                    self.nngraph(**self._dna_nngraph)

            tower_devices = [] + self['devices']
            if self['sep_device']: tower_devices = tower_devices[1:] # trim SEP
            for dev in tower_devices:
                self._nwwlog.debug(f'NEModel builds FWD graph @device: {dev}')
                with tf.device(dev):
                    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                        self._gFWD.append(self.nngraph(**self._dna_nngraph))

            fwd_graph_return_dict = self._gFWD[0]
            self._nwwlog.debug(f'dictionary keys returned by fwd_func ({self.nngraph.__name__}): {fwd_graph_return_dict.keys()}')

            self.update(fwd_graph_return_dict) # update self with fwd_graph_return_dict

            # get FWD variables returned by fwd_func (4 saver)
            train_vars = [] # variables to train
            saver_vars = {} # dict of variables to save
            for key in self.get_all_fields():
                if 'var' in key.lower():
                    if key =='train_vars':
                        train_vars = self[key]
                        if type(train_vars) is not list: train_vars = [train_vars]
                    else:
                        if type(self[key]) is not list: saver_vars[key] = [self[key]]
                        else:                           saver_vars[key] = self[key]
            all_vars = tf.global_variables()

            # there are returned variables >> assert there are all variables returned in lists
            if saver_vars:
                all_vars_returned = []
                for key in saver_vars: all_vars_returned += saver_vars[key]
                there_are_all = True
                for var in all_vars:
                    if var not in all_vars_returned:
                        print(f' *** variable {var.name} not returned by fwd_func')
                        there_are_all = False
                assert there_are_all, 'ERR: there are some variables not returned by fwd_func in lists!'

            else: saver_vars['fwd_vars'] = all_vars # put all


            self._nwwlog.debug('NEModel variables to save from fwd_func:')
            for key in sorted(list(saver_vars.keys())):
                    varList = saver_vars[key]
                    if varList: self._nwwlog.log(5, f' ### vars @{key} - num: {len(varList)}, floats: {short_scin(num_var_floats(varList))} ({varList[0].device})')
                    else: self._nwwlog.log(5, ' ### no vars')
                    self._nwwlog.log(5, log_vars(varList))

            if 'loss' not in self: self._nwwlog.warning('NEModel: there is no loss in FWD graph, OPT graph wont be build!')
            if not self['opt_func']: self._nwwlog.warning(f'\nNEModel: OPT graph wont be build since opt_func is not given')

            # build optimization graph
            if self['opt_func'] and 'loss' in self:
                self._nwwlog.debug(f'Preparing OPT part with {self["opt_class"]}')

                # select trainable variables for OPT
                all_tvars = tf.trainable_variables()
                if train_vars:
                    # check if all train_vars are trainable:
                    for var in train_vars:
                        if var not in all_tvars:
                            self._nwwlog.warning(f'variable {var.name} is not trainable but is in train_vars, please check the graph!')
                else:
                    for key in saver_vars:
                        for var in saver_vars[key]:
                            if var in all_tvars:
                                train_vars.append(var)
                    assert train_vars, 'ERR: there are no trainable variables at the graph!'
                # log train_vars
                self._nwwlog.debug('NEModel trainable variables:')
                self._nwwlog.debug(f' ### train_vars: {len(train_vars)} floats: {short_scin(num_var_floats(train_vars))}')
                self._nwwlog.log(5, log_vars(train_vars))

                # build gradients for towers
                for ix in range(len(self._gFWD)):
                    tower = self._gFWD[ix]
                    tower['gradients'] = tf.gradients(
                        ys=                             tower['loss'],
                        xs=                             train_vars,
                        colocate_gradients_with_ops=    not self['collocate_GWO']) # TF default is False >> calculates gradients where OPS, for True >> where train_vars

                    # log gradients

                    # None_as_gradient case
                    device = 'UNKNOWN'
                    for t in tower['gradients']:
                        if t is not None:
                            device = t.device
                            break

                    self._nwwlog.debug(f' > gradients for {ix} tower got {len(tower["gradients"])} tensors ({device})')

                    self._nwwlog.log(5, 'NEModel variables and their gradients:')
                    for gix in range(len(tower['gradients'])):
                        grad = tower['gradients'][gix]
                        var = train_vars[gix]
                        self._nwwlog.log(5, f'{var} {var.device}')
                        self._nwwlog.log(5, f' > {grad}') # grad as a tensor displays device when printed (unless collocated with OP!)

                self['gradients'] = self._gFWD[0]['gradients']

                # check for None @ gradients
                none_grads = 0
                for grad in self['gradients']:
                    if grad is None: none_grads += 1
                if none_grads: self._nwwlog.warning(f'There are None gradients: {none_grads}/{len(self["gradients"])}, some trainVars may be unrelated to loss, please check the graph!')

                # average gradients
                if len(self['devices']) > 1:

                    self._nwwlog.debug(f'NEModel builds gradients averaging graph with device {self["devices"][0]} for {len(self._gFWD)} towers')
                    with tf.device(self["devices"][0]):
                        towerGrads = [tower['gradients'] for tower in self._gFWD]
                        avgGrads = []
                        for mGrads in zip(*towerGrads):
                            grads = []
                            for grad in mGrads:
                                if grad is not None: # None for variables not used while training now...
                                    expandedG = tf.expand_dims(input=grad, axis=-1)
                                    grads.append(expandedG)
                            if grads:
                                grad = tf.concat(values=grads, axis=-1)
                                grad = tf.reduce_mean(input_tensor=grad, axis=-1)
                                avgGrads.append(grad)
                            else: avgGrads.append(None)

                        self['gradients'] = avgGrads # update with averaged gradients
                        self._nwwlog.debug(f' > NEModel averaged gradients ({self["gradients"][0].device})')

                # finally build graph from elements
                with tf.variable_scope('OPT', reuse=tf.AUTO_REUSE):

                    self._nwwlog.debug(f'Building OPT graph for {self.name} model @device: {self["devices"][0]}')

                    with tf.device(self['devices'][0]):

                        opt_graph_return_dict = self['opt_func'](
                            train_vars=     train_vars,
                            gradients=      self['gradients'],
                            **self._dna_opt_func)
                        self._nwwlog.debug(f'dictionary keys returned by opt_func ({self["opt_func"].__name__}): {opt_graph_return_dict.keys()}')

                        self.update(opt_graph_return_dict)  # update self with opt_graph_return_dict

                        saver_vars['opt_vars'] = self['opt_vars']

        # create session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self._session = tf.Session(
            graph=  self._graph,
            config= config)

        # create saver & load
        sv_keys = list(saver_vars.keys())
        for key in sv_keys:
            if not saver_vars[key]: saver_vars.pop(key) # remove keys with no variables (corner case, for proper saver)
        # add saver then load
        self._saver = MultiSaver(
            model_name= self.name,
            vars=       saver_vars,
            save_TFD=   self.save_topdir,
            savers=     self['savers_names'],
            session=    self._session,
            logger=     get_hi_child(self._nwwlog, 'MultiSaver'))
        if self['load_saver']: self.load_ckpt()

    def __call__(self, feed_dict:dict, fetch:List[str]) -> dict:
        out = self._session.run(feed_dict=feed_dict, fetches=[self[n] for n in fetch])
        return {n: e for n,e in zip(fetch,out)}

    def backward(self, feed_dict:dict, fetch:List[str]) -> dict:
        return self.__call__(feed_dict=feed_dict, fetch=fetch)

    # *********************************************************************************************** load / save / copy

    # reloads model checkpoint, updates baseLR
    def load_ckpt(self) -> None:
        saver = None if type(self['load_saver']) is bool else self['load_saver']
        self._saver.load(saver=saver)
        if 'baseLR' in self: self.update_baseLR(self['baseLR'])

    # saves model checkpoint
    def save_ckpt(self) -> None:
        assert not self['read_only'], f'ERR: cannot save NEModel checkpoint {self.name} while model is readonly!'
        self._saver.save()

    @classmethod
    def copy_checkpoint(
            cls,
            name_src: str,
            name_trg: str,
            save_topdir_src: Optional[str]= None,
            save_topdir_trg: Optional[str]= None):

        if not save_topdir_src: save_topdir_src = cls.SAVE_TOPDIR
        if not save_topdir_trg: save_topdir_trg = save_topdir_src

        nm_SFD = f'{save_topdir_src}/{name_src}'
        ckptL = [cfd for cfd in os.listdir(nm_SFD) if os.path.isdir(os.path.join(nm_SFD, cfd))]
        if 'opt_vars' in ckptL: ckptL.remove('opt_vars')
        for ckpt in ckptL:
            mrg_ckpts(
                ckptA=          ckpt,
                ckptA_FD=       nm_SFD,
                ckptB=          None,
                ckptB_FD=       None,
                ckptM=          ckpt,
                ckptM_FD=       f'{save_topdir_trg}/{name_trg}',
                replace_scope=  name_trg)

    # *************************************************************************************************************** GX

    @classmethod
    def gx_ckpt(
            cls,
            name_A: str,                        # name parent A
            name_B: str,                        # name parent B
            name_child: str,                    # name child
            save_topdir_A: Optional[str]=       None,
            save_topdir_B: Optional[str]=       None,
            save_topdir_child: Optional[str]=   None,
            ratio: float=                       0.5,
            noise: float=                       0.03):

        if not save_topdir_A: save_topdir_A = cls.SAVE_TOPDIR
        if not save_topdir_B: save_topdir_B = save_topdir_A
        if not save_topdir_child: save_topdir_child = save_topdir_A

        mfd = f'{save_topdir_A}/{name_A}'
        ckptL = [dI for dI in os.listdir(mfd) if os.path.isdir(os.path.join(mfd,dI))]
        if 'opt_vars' in ckptL: ckptL.remove('opt_vars')

        for ckpt in ckptL:
            mrg_ckpts(
                ckptA=          ckpt,
                ckptA_FD=       f'{save_topdir_A}/{name_A}/',
                ckptB=          ckpt,
                ckptB_FD=       f'{save_topdir_B}/{name_B}/',
                ckptM=          ckpt,
                ckptM_FD=       f'{save_topdir_child}/{name_child}/',
                replace_scope=  name_child,
                ratio=          ratio,
                noise=          noise)

    # ***************************************************************************************************** train / test

    # builds feed dict from given batch of data
    def build_feed(self, batch: dict, train=True) -> dict:
        self._nwwlog.warning('NEModel.build_feed() should be overridden!')
        return {}

    # TODO: refactor according to MOTorch train concept
    #  - load_data()
    def run_train(
            self,
            n_batches: Optional[int]=   None,
            test_freq=                  100,
            mov_avg_factor=             0.1,
            save_max=                   True,
            use_F1=                     True, # TODO: use
            **kwargs) -> float:

        if not self._batcher: raise NEModelException('NEModel has not been given data for training, use load_data() or give it while training!')

        self._nwwlog.info(f'{self.name} - training starts [acc / F1 / loss]')
        self._nwwlog.info(f'data sizes (TR,VL,TS) samples: {self._batcher.get_data_size()}')

        if n_batches is None: n_batches = self['n_batches']  # take default
        self._nwwlog.info(f'batch size:             {self["batch_size"]}')
        self._nwwlog.info(f'train for num_batches:  {n_batches}')

        batch_IX = 0
        tr_accL = []
        tr_f1L = []
        tr_lssL = []

        score_name = 'F1' if use_F1 else 'acc'
        ts_score_max = 0  # test score (acc or F1) max
        ts_score_all_results = []  # test score all results
        ts_score_mav = MovAvg(mov_avg_factor)  # test score (acc or F1) moving average

        ts_bIX = [bIX for bIX in range(n_batches+1) if not bIX % test_freq] # batch indexes when test will be performed
        assert ts_bIX, 'ERR: model SHOULD BE tested while training!'
        ten_factor = int(0.1*len(ts_bIX)) # number of tests for last 10% of training
        if ten_factor < 1: ten_factor = 1 # we need at least one result
        if self['hpmser_mode']: ts_bIX = ts_bIX[-ten_factor:]

        while batch_IX < n_batches:

            batch = self._batcher.get_batch()
            feed = self.build_feed(batch)
            out = self(feed_dict=feed, fetch=['optimizer','loss','acc','f1','gg_norm','gg_avt_norm'])
            out.pop('optimizer')

            batch_IX += 1
            self['train_batch_IX'] += 1

            if self['do_TB']:
                for k in out:
                    self.log_TB(value=out[k], tag=f'tr/{k}', step=self['train_batch_IX'])
            tr_accL.append(out['acc'])
            tr_f1L.append(out['f1'])
            tr_lssL.append(out['loss'])

            if batch_IX in ts_bIX:
                ts_acc, ts_f1, ts_loss = self.run_test()

                ts_score = ts_f1 if use_F1 else ts_acc
                if ts_score is not None:
                    ts_score_all_results.append(ts_score)
                if self['do_TB']:
                    self.log_TB(value=ts_loss,                      tag='ts/loss',              step=self['train_batch_IX'])
                    self.log_TB(value=ts_acc,                       tag='ts/acc',               step=self['train_batch_IX'])
                    self.log_TB(value=ts_f1,                        tag='ts/F1',                step=self['train_batch_IX'])
                    self.log_TB(value=ts_score_mav.upd(ts_score),   tag=f'ts/{score_name}_mav', step=self['train_batch_IX'])

                tr_acc_nfo = f'{100*sum(tr_accL)/test_freq:.1f}'
                tr_f1_nfo =  f'{100*sum(tr_f1L)/test_freq:.1f}'
                ts_acc_nfo = f'{100*ts_acc:.1f}'
                ts_f1_nfo = f'{100*ts_f1:.1f}'
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

        return ts_score_wval

    def run_test(self, **kwargs) -> Tuple[Optional[float], Optional[float], float]:

        if not self._batcher: raise NEModelException('NEModel has not been given data for testing, use load_data() or give it while testing!')

        batches = self._batcher.get_TS_batches()
        lossL = []
        accL = []
        f1L = []
        for batch in batches:
            feed = self.build_feed(batch)
            out = self(feed_dict=feed, fetch=['loss','acc','f1'])
            lossL.append(out['loss'])
            accL.append(out['acc'])
            f1L.append(out['f1'])

        return sum(accL)/len(accL), sum(f1L)/len(f1L), sum(lossL)/len(lossL)

    @property
    def gFWD(self):
        return self._gFWD

    @property
    def session(self):
        return self._session

    # updates baseLR in graph - but not saves it to the checkpoint
    def update_baseLR(self, lr: Optional[float]) -> None:
        if 'baseLR_var' not in self:
            self._nwwlog.warning('NEModel: There is no LR variable in graph to update')
        else:
            if lr is not None:
                old = self['baseLR']
                self['baseLR'] = lr
                self._nwwlog.debug(f'NEModel {self.name} updated baseLR from {old} to {self["baseLR"]}')
            self._session.run(tf.assign(ref=self['baseLR_var'], value=self['baseLR']))
            self._nwwlog.debug(f'NEModel {self.name} updated baseLR_var (graph variable) with baseLR: {self["baseLR"]}')

    def __str__(self):
        # TODO: add some NEModel-specific output (name, graph info, weights)
        return ParaSave.__str__(self)