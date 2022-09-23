"""

 2019 (c) piteren

 NEModel class:

    Builds complete NN model object with implemented many features to effectively maintain components and procedures like:
     graph, GPU resources, save & load (serialize), train/test baseline, TB, GX ..and many others.

    fwd_func:
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

    NEModel manages params of self, FWD & OPT graph functions. Those params may come from different sources**:
        - NEModel class init defaults
        - OPT graph defaults
        - FWD graph defaults
        - params saved in folder
        - given kwargs (DNA)
        ** name - must be given to NEModel
        ** verb - similar (when not given is always set to 0)

    NEModel keeps all params in self as a Subscriptable. Objects like graph, session, saver, tensors, placeholders, etc.
    created by NEModel or returned by graphs functions are also kept in self fields and may be easily accessed.

 NEModel class implements:
    - one folder for all model data: DNA and checkpoints (subfolder of save_TFD named with model name)
    - parameters management with proper resolution
    - logger (txt file saved into the model folder)
    - GPU automated management with multi-GPU training on towers (places model graph elements across available devices)
    - builds optimization (OPT) graph with default OPT function
        - calculates gradients for every tower >> averages them
        - AVT gradient clipping and scaled LR (warmup, annealing)
    - builds forward (FWD) graph with given function
    - gots exemplary FWD graph function
    - MultiSaver (with option of saving/loading sub-lists of variables (separate checkpoints with version management)
    - inits session, TB writer, MultiSaver loads or inits variables
    - ParaSave interface + with GXable interface, implements GX on checkpoints
    - baseline methods for training / testing
    - sanity check of many graph elements and dependencies
"""

import numpy as np
import os
from typing import Optional, Callable
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pypaq.lipytools.logger import set_logger
from pypaq.lipytools.little_methods import short_scin, stamp, get_params, get_func_dna
from pypaq.lipytools.moving_average import MovAvg
from pypaq.mpython.devices import mask_cuda, get_devices
from pypaq.mpython.mpdecor import proc_wait
from pypaq.pms.parasave import ParaSave
from pypaq.neuralmess.get_tf import tf
from pypaq.neuralmess.base_elements import num_var_floats, lr_scaler, gc_loss_reductor, log_vars, mrg_ckpts, TBwr
from pypaq.neuralmess.layers import lay_dense
from pypaq.neuralmess.multi_saver import MultiSaver
from pypaq.comoneural.batcher import Batcher


# restricted keys for fwd_func DNA and return DNA (if they appear in kwargs, should be named exactly like below)
SPEC_KEYS = [
    'train_vars',   # list of variables to train (may be returned, otherwise all trainable are taken)
    'opt_vars',     # list of variables returned by opt_func
    'loss',         # loss
    'acc',          # accuracy
    'f1',           # F1
]

# defaults below may be given with NEModelDUO kwargs or overridden by fwd_graph attributes
NEMODEL_DEFAULTS = {
    'seed':                 12321,                      # seed for TF and numpy
    'devices':              -1,                         # : DevicesParam (check pypaq.mpython.devices)
        # training
    'batch_size':           64,                         # training batch size
    'n_batches':            1000,                       # default length of training
        # other
    'save_topdir':          '_models',                  # top folder of model save
    'save_fn_pfx':          'nemodel_dna',              # dna filename prefix
    'hpmser_mode':          False,                      # it will set model to be read_only and quiet when running with hpmser
    'savers_names':         (None,),                    # names of savers for MultiSaver
    'load_saver':           True,                       # Optional[bool or str] for None/False does not load, for True loads default
    'read_only':            False,                      # sets model to be read only - wont save anything (wont even create self.model_dir)
    'do_logfile':           True,                       # enables saving log file in self.model_dir
    'do_TB':                True,                       # runs TensorBard, saves in self.model_dir
    'silent_TF_warnings':   False,                      # turns off TF warnings
    'sep_device':           True,                       # separate first device for variables, gradients_avg, optimizer (otherwise those ar placed on the first FWD calculations tower)
    'collocate_GWO':        False,                      # collocates gradient calculations with tf.OPs (gradients are calculated on every tower with its operations, but remember that vars are on one device...) (otherwise with first FWD calculations tower)
}


# default FWD function (forward graph), it is given as an exemplary implementation
def fwd_graph(
        name: str,
        seq_len: int,
        emb_num: int=   50,
        emb_width: int= 100,
        n_labels: int=  2,
        seed: int=      321,
        iLR=            0.003):
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
    return {
        'in_PH':    in_PH,
        'emb':      emb,
        'logits':   logits,
        'loss':     loss}

# default OPT function (optimization graph), should be used in most scenarios
def opt_graph(
        train_vars,
        gradients,
        opt_class=          tf.train.AdamOptimizer, # default optimizer, other examples: tf.train.GradientDescentOptimizer, partial(tf.train.AdamOptimizer, beta1=0.7, beta2=0.7)
            # LR management (parameters of LR warmup and annealing)
        iLR=                3e-4,                   # initial learning rate (base init)
        warm_up=            None,
        ann_base=           None,
        ann_step=           1,
        n_wup_off: float=   1.0,
            # gradients clipping parameters
        avt_SVal=           1,
        avt_window=         100,
        avt_max_upd=        1.5,
        do_clip=            False,
        verb=               0):

    g_step = tf.get_variable(  # global step variable
        name=           'g_step',
        shape=          [],
        trainable=      False,
        initializer=    tf.constant_initializer(0),
        dtype=          tf.int32)

    iLR_var = tf.get_variable(  # base LR variable
        name=           'iLR',
        shape=          [],
        trainable=      False,
        initializer=    tf.constant_initializer(iLR),
        dtype=          tf.float32)

    scaled_LR = lr_scaler(
        iLR=            iLR_var,
        g_step=         g_step,
        warm_up=        warm_up,
        ann_base=       ann_base,
        ann_step=       ann_step,
        n_wup_off=      n_wup_off,
        verb=           verb)['scaled_LR']

    # updates with: optimizer, gg_norm, gg_avt_norm
    loss_reductorD = gc_loss_reductor(
        optimizer=      opt_class(learning_rate=scaled_LR),
        vars=           train_vars,
        g_step=         g_step,
        gradients=      gradients,
        avt_SVal=       avt_SVal,
        avt_window=     avt_window,
        avt_max_upd=    avt_max_upd,
        do_clip=        do_clip,
        verb=           verb)

    # select OPT vars
    opt_vars = tf.global_variables(scope=tf.get_variable_scope().name)
    if verb>0:
        print(f' ### opt_vars: {len(opt_vars)} floats: {short_scin(num_var_floats(opt_vars))} ({opt_vars[0].device})')
        if verb>1: log_vars(opt_vars)

    rd = {}
    rd.update({
        'g_step':       g_step,
        'iLR_var':      iLR_var,
        'scaled_LR':    scaled_LR,
        'opt_vars':     opt_vars})
    rd.update(loss_reductorD)
    return rd


# NEModel Base class, implements most features (but not saving)
class NEModel(ParaSave):

    def __init__(
            self,
            name: str,
            name_timestamp=                 False,      # adds timestamp to model name
            fwd_func: Optional[Callable]=   None,       # function building graph (from inputs to loss) - may be not given if model already saved
            opt_func: Optional[Callable]=   opt_graph,  # default function building optimization (OPT) graph (from train_vars & gradients to optimizer)
            save_topdir=                    NEMODEL_DEFAULTS['save_topdir'],
            save_fn_pfx=                    NEMODEL_DEFAULTS['save_fn_pfx'],
            verb=                           0,
            **kwargs):                                  # here go params of FWD & OPT functions

        if name_timestamp: name += f'.{stamp()}'
        if verb>0: print(f'\n *** NEModel {name} (type: {type(self).__name__}) *** initializes...')

        # ************************************************************************* manage (resolve) DNA & init ParaSave

        # load dna from folder
        dna_saved = ParaSave.load_dna(
            name=           name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx)

        assert fwd_func or 'fwd_func' in dna_saved, 'ERR: cannot continue: fwd_func was not given and model has not been saved before.'

        _opt_func_params_defaults = get_params(opt_func)['with_defaults'] if opt_func else {}
        _fwd_func_params_defaults = get_params(fwd_func)['with_defaults']  # get fwd_func defaults

        dna = {
            'fwd_func': fwd_func,
            'opt_func': opt_func}
        dna.update(NEMODEL_DEFAULTS)
        dna.update(_opt_func_params_defaults)
        dna.update(_fwd_func_params_defaults)
        dna.update(dna_saved)
        dna.update(kwargs)                  # update with kwargs (params of FWD & OPT) given NOW by user
        dna.update({
            'name':         name,
            'save_topdir':  save_topdir,
            'save_fn_pfx':  save_fn_pfx,
            'verb':         verb})

        ParaSave.__init__(self, lock_managed_params=True, **dna)
        self.check_params_sim(SPEC_KEYS + list(NEMODEL_DEFAULTS.keys())) # safety check

        # hpmser_mode - early override
        if self['hpmser_mode']:
            self.verb = 0
            self['read_only'] = True

        # read only - early override
        if self['read_only']:
            self['do_logfile'] = False
            self['do_TB'] = False

        if self['silent_TF_warnings']:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            warnings.filterwarnings('ignore')

        self.model_dir = f'{self.save_topdir}/{self.name}'
        if self.verb>0: print(f' > NEModel dir: {self.model_dir}{" read only mode!" if self["read_only"] else ""}')

        # set logfile
        if self['do_logfile']:
            set_logger(
                log_folder=     self.model_dir,
                custom_name=    self.name,
                verb=           self.verb)

        if self.verb>0:

            not_used_kwargs = {}
            for k in kwargs:
                if k not in get_func_dna(self['opt_func'], dna) and k not in get_func_dna(self['fwd_func'], dna):
                    not_used_kwargs[k] = kwargs[k]

            print(f'\n > NEModel DNA sources:')
            print(f' >> NEMODEL_DEFAULTS:                   {NEMODEL_DEFAULTS}')
            print(f' >> fwd_func defaults:                  {_fwd_func_params_defaults}')
            print(f' >> opt_func defaults:                  {_opt_func_params_defaults}')
            print(f' >> DNA saved:                          {dna_saved}')
            print(f' >> given kwargs:                       {kwargs}')
            print(f' >> NEModel kwargs not used by graphs : {not_used_kwargs}')
            print(f' NEModel complete DNA:                  {dna}')

        self.__manage_devices()

        # set seed
        tf.set_random_seed(self['seed'])
        np.random.seed(self['seed'])

        self._gFWD = [] # list of dicts of all FWD graphs (from all devices)
        self._graph = None
        saver_vars = self.__build_graph()
        if self.verb>0: print(f'{self.name} (NEModel) graph built!')
        if self.verb>1: print(f' > self point:\n{self.get_point()}')
        if self.verb>1: print(f' > self all fields:\n{self.get_all_fields()}')

        # create session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self._session = tf.Session(
            graph=  self._graph,
            config= config)

        self.__TBwr = TBwr(logdir=self.model_dir)  # TensorBoard writer

        # create saver & load
        sv_keys = list(saver_vars.keys())
        for key in sv_keys:
            if not saver_vars[key]: saver_vars.pop(key) # remove keys with no variables (corner case, for proper saver)
        # add saver then load
        self.__saver = MultiSaver(
            model_name= self.name,
            vars=       saver_vars,
            save_TFD=   self.save_topdir,
            savers=     self['savers_names'],
            session=    self._session,
            verb=       self.verb)
        if self['load_saver']: self.load_ckpt()

        self._model_data = None
        self._batcher = None

        if self.verb>0: print(f'\n > NEModel init finished..')

    # sets CPU / GPU devices for NEModel
    def __manage_devices(self):

        self['devices'] = get_devices(self['devices'], verb=self.verb)

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

        if self.verb>0: print(f' > masking GPU devices: {ids}')
        mask_cuda(ids) # mask GPU devices

        # report devices
        if self.verb>0:
            print()
            if len(self['devices'])==1:
                if 'CPU' in self['devices'][0]: print(f'NEModel builds CPU device setup')
                else:                           print(f'NEModel builds single-GPU setup')
            else:                               print(f'NEModel builds multi-dev setup for {len(self["devices"])} devices')

        if len(self['devices'])<3: self['sep_device'] = False # SEP is available for 3 or more devices

    # builds graph (FWD & OPT) and manages surroundings
    def __build_graph(self) -> dict:

        # build FWD graph(s) >> manage variables >> build OPT graph
        self._graph = tf.Graph()
        with self._graph.as_default():

            if self.verb>0: print(f'\nNEModel set TF & NP seed to {self["seed"]}')

            fwd_func_dna = get_func_dna(self['fwd_func'], self.get_point())

            # builds graph @SEP, this graph wont be run, it is only needed to place variables, if not vars_sep >> variables will be placed with first tower
            if self['sep_device']:
                if self.verb>0: print(f'\nNEModel places VARs on {self["devices"][0]}...')
                with tf.device(self['devices'][0]):
                    self['fwd_func'](**fwd_func_dna)

            tower_devices = [] + self['devices']
            if self['sep_device']: tower_devices = tower_devices[1:] # trim SEP
            for dev in tower_devices:
                if self.verb>0: print(f'\nNEModel builds FWD graph @device: {dev}')
                with tf.device(dev):
                    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                        self._gFWD.append(self['fwd_func'](**fwd_func_dna))

            fwd_graph_return_dict = self._gFWD[0]
            if self.verb>0: print(f'dictionary keys returned by fwd_func ({self["fwd_func"].__name__}): {fwd_graph_return_dict.keys()}')

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

            if self.verb>0:
                print('\nNEModel variables to save from fwd_func:')
                for key in sorted(list(saver_vars.keys())):
                    varList = saver_vars[key]
                    if varList: print(f' ### vars @{key} - num: {len(varList)}, floats: {short_scin(num_var_floats(varList))} ({varList[0].device})')
                    else: print(' ### no vars')
                    if self.verb>1: log_vars(varList)

            if 'loss' not in self: warnings.warn('NEModel: there is no loss in FWD graph, OPT graph wont be build!')
            if not self['opt_func']: print(f'\nNEModel: OPT graph wont be build since opt_func is not given')

            # build optimization graph
            if self['opt_func'] and 'loss' in self:
                if self.verb>0: print(f'\nPreparing OPT part with {self["opt_class"]}')

                # select trainable variables for OPT
                all_tvars = tf.trainable_variables()
                if train_vars:
                    # check if all train_vars are trainable:
                    for var in train_vars:
                        if var not in all_tvars:
                            if self.verb>0: print(f'variable {var.name} is not trainable but is in train_vars, please check the graph!')
                else:
                    for key in saver_vars:
                        for var in saver_vars[key]:
                            if var in all_tvars:
                                train_vars.append(var)
                    assert train_vars, 'ERR: there are no trainable variables at the graph!'
                # log train_vars
                if self.verb>0:
                    print('\nNEModel trainable variables:')
                    print(f' ### train_vars: {len(train_vars)} floats: {short_scin(num_var_floats(train_vars))}')
                    if self.verb>1: log_vars(train_vars)

                # build gradients for towers
                for ix in range(len(self._gFWD)):
                    tower = self._gFWD[ix]
                    tower['gradients'] = tf.gradients(
                        ys=                             tower['loss'],
                        xs=                             train_vars,
                        colocate_gradients_with_ops=    not self['collocate_GWO']) # TF default is False >> calculates gradients where OPS, for True >> where train_vars

                    # log gradients
                    if self.verb>0:

                        # None_as_gradient case
                        device = 'UNKNOWN'
                        for t in tower['gradients']:
                            if t is not None:
                                device = t.device
                                break

                        print(f' > gradients for {ix} tower got {len(tower["gradients"])} tensors ({device})')
                        if self.verb>1:
                            print('NEModel variables and their gradients:')
                            for gix in range(len(tower['gradients'])):
                                grad = tower['gradients'][gix]
                                var = train_vars[gix]
                                print(var, var.device)
                                print(f' > {grad}') # grad as a tensor displays device when printed (unless collocated with OP!)

                self['gradients'] = self._gFWD[0]['gradients']

                # check for None @ gradients
                none_grads = 0
                for grad in self['gradients']:
                    if grad is None: none_grads += 1
                if none_grads and self.verb>0:
                    print(f'There are None gradients: {none_grads}/{len(self["gradients"])}, some trainVars may be unrelated to loss, please check the graph!')

                # average gradients
                if len(self['devices']) > 1:

                    if self.verb>0: print(f'\nNEModel builds gradients averaging graph with device {self["devices"][0]} for {len(self._gFWD)} towers')
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
                        if self.verb>0: print(f' > NEModel averaged gradients ({self["gradients"][0].device})')

                # finally build graph from elements
                with tf.variable_scope('OPT', reuse=tf.AUTO_REUSE):

                    if self.verb>0: print(f'\nBuilding OPT graph for {self.name} model @device: {self["devices"][0]}')

                    opt_func_dna = get_func_dna(self['opt_func'], self.get_point())

                    with tf.device(self['devices'][0]):

                        opt_graph_return_dict = self['opt_func'](
                            train_vars=     train_vars,
                            gradients=      self['gradients'],
                            **opt_func_dna)
                        if self.verb>0: print(f'dictionary keys returned by opt_func ({self["opt_func"].__name__}): {opt_graph_return_dict.keys()}')

                        self.update(opt_graph_return_dict)  # update self with opt_graph_return_dict

                        saver_vars['opt_vars'] = self['opt_vars']

        return saver_vars

    # reloads model checkpoint, updates iLR
    def load_ckpt(self):
        saver = None if type(self['load_saver']) is bool else self['load_saver']
        self.__saver.load(saver=saver)
        if 'iLR' in self: self.update_LR(self['iLR'])

    # saves model checkpoint
    def save_ckpt(self):
        assert not self['read_only'], f'ERR: cannot save NEModel checkpoint {self.name} while model is readonly!'
        self.__saver.save()

    # updates base LR (iLR) in graph - but not saves it to the checkpoint
    def update_LR(self, lr: Optional):
        if 'iLR_var' not in self:
            if self.verb>1: print('NEModel: There is no LR variable in graph to update')
        else:
            if lr is not None:
                old = self['iLR']
                self['iLR'] = lr
                if self.verb>1: print(f'NEModel {self.name} updated iLR from {old} to {self["iLR"]}')
            self._session.run(tf.assign(ref=self['iLR_var'], value=self['iLR']))
            if self.verb>1: print(f'NEModel {self.name} updated iLR_var (graph variable) with iLR: {self["iLR"]}')

    # logs value to TB
    def log_TB(
            self,
            value,
            tag: str,
            step: int):
        if self['do_TB']: self.__TBwr.add(value=value, tag=tag, step=step)
        else: warnings.warn(f'NEModel {self.name} cannot log TensorBoard since do_TB flag is False!')

    # GX for two NEModel checkpoints
    @staticmethod
    def gx_ckpt(
            name_A: str,                        # name parent A
            name_B: str,                        # name parent B
            name_child: str,                    # name child
            folder_A: str=                  NEMODEL_DEFAULTS['save_topdir'],
            folder_B: Optional[str]=        None,
            folder_child: Optional[str]=    None,
            ratio: float=                   0.5,
            noise: float=                   0.03):

        if not folder_B: folder_B = folder_A
        if not folder_child: folder_child = folder_A

        mfd = f'{folder_A}/{name_A}'
        ckptL = [dI for dI in os.listdir(mfd) if os.path.isdir(os.path.join(mfd,dI))]
        if 'opt_vars' in ckptL: ckptL.remove('opt_vars')

        for ckpt in ckptL:
            mrg_ckpts(
                ckptA=          ckpt,
                ckptA_FD=       f'{folder_A}/{name_A}/',
                ckptB=          ckpt,
                ckptB_FD=       f'{folder_B}/{name_B}/',
                ckptM=          ckpt,
                ckptM_FD=       f'{folder_child}/{name_child}/',
                replace_scope=  name_child,
                ratio=          ratio,
                noise=          noise)

    # **************************************************************************************** baseline training methods

    # loads model data for training, dict should have at least 'train':{} for Batcher
    def load_model_data(self) -> dict:
        warnings.warn('NEModel.load_model_data() should be overridden!')
        return {}

    # pre (before) training method - may be overridden
    def pre_train(self):
        self._model_data = self.load_model_data()
        self._batcher = Batcher(
            data_TR=        self._model_data['train'],
            data_VL=        self._model_data['valid'] if 'valid' in self._model_data else None,
            data_TS=        self._model_data['test'] if 'test' in self._model_data else None,
            batch_size=     self['batch_size'],
            batching_type=  'random_cov',
            verb=           self.verb)

    # builds feed dict from given batch of data
    def build_feed(self, batch: dict, train=True) -> dict:
        warnings.warn('NEModel.build_feed() should be overridden!')
        return {}

    # TODO: refactor according to MOTorch train concept
    #  - load_data()
    # training method, saves max
    def train(
            self,
            n_batches: Optional[int]=   None,
            test_freq=                  100,    # number of batches between tests, model SHOULD BE tested while training
            mov_avg_factor=             0.1,
            save=                       True    # allows to save model while training
    ):

        self.pre_train()

        if self.verb>0: print(f'{self.name} - training starts')
        if n_batches is None: n_batches = self['n_batches']  # take default
        batch_IX = 0
        tr_lssL = []
        tr_accL = []
        ts_acc_max = 0
        ts_acc_mav = MovAvg(mov_avg_factor)

        ts_results = []
        ts_bIX = [bIX for bIX in range(n_batches+1) if not bIX % test_freq] # batch indexes when test will be performed
        assert ts_bIX, 'ERR: model SHOULD BE tested while training!'
        ten_factor = int(0.1*len(ts_bIX)) # number of tests for last 10% of training
        if ten_factor < 1: ten_factor = 1 # we need at least one result
        if self['hpmser_mode']: ts_bIX = ts_bIX[-ten_factor:]

        while batch_IX < n_batches:
            batch_IX += 1
            batch = self._batcher.get_batch()

            feed = self.build_feed(batch)
            fetches = self['optimizer']
            if self['do_TB'] or self.verb>0: fetches = [self['optimizer'], self['loss'], self['acc'], self['gg_norm'], self['gg_avt_norm']]

            run_out = self._session.run(fetches, feed)

            if self['do_TB'] or self.verb>0:
                _, loss, acc, gg_norm, gg_avt_norm = run_out
                if self['do_TB']:
                    self.log_TB(value=loss,        tag='tr/loss',    step=batch_IX)
                    self.log_TB(value=acc,         tag='tr/acc',     step=batch_IX)
                    self.log_TB(value=gg_norm,     tag='tr/gn',      step=batch_IX)
                    self.log_TB(value=gg_avt_norm, tag='tr/gn_avt',  step=batch_IX)
                tr_lssL.append(loss)
                tr_accL.append(acc)

            if batch_IX in ts_bIX:
                ts_acc, ts_loss = self.test()
                acc_mav = ts_acc_mav.upd(ts_acc)
                ts_results.append(ts_acc)
                if self['do_TB']:
                    self.log_TB(value=ts_loss, tag='ts/loss',    step=batch_IX)
                    self.log_TB(value=ts_acc,  tag='ts/acc',     step=batch_IX)
                    self.log_TB(value=acc_mav, tag='ts/acc_mav', step=batch_IX)
                if self.verb>0: print(f'{batch_IX:5d} TR: {100*sum(tr_accL)/test_freq:.1f} / {sum(tr_lssL)/test_freq:.3f} -- TS: {100*ts_acc:.1f} / {ts_loss:.3f}')
                tr_lssL = []
                tr_accL = []

                if ts_acc > ts_acc_max:
                    ts_acc_max = ts_acc
                    if not self['read_only'] and save: self.save_ckpt() # model is saved for max_ts_acc

        # weighted test value for last 10% test results
        ts_results = ts_results[-ten_factor:]
        ts_wval = 0
        weight = 1
        sum_weight = 0
        for tr in ts_results:
            ts_wval += tr*weight
            sum_weight += weight
            weight += 1
        ts_wval /= sum_weight
        if self['do_TB']: self.log_TB(value=ts_wval, tag='ts/ts_wval', step=batch_IX)
        if self.verb>0:
            print(f'model {self.name} finished training')
            print(f' > test_acc_max: {ts_acc_max:.4f}')
            print(f' > test_wval:    {ts_wval:.4f}')

        return ts_wval

    def test(self):
        batches = self._batcher.get_TS_batches()
        acc_loss = []
        acc_acc = []
        for batch in batches:
            feed = self.build_feed(batch, train=False)
            fetches = [self['loss'], self['acc']]
            loss, acc = self._session.run(fetches, feed)
            acc_loss.append(loss)
            acc_acc.append(acc)
        return sum(acc_acc)/len(acc_acc), sum(acc_loss)/len(acc_loss)

    # copies full NEModel folder (DNA & checkpoints)
    @staticmethod
    def copy_saved(
            name_src: str,
            name_trg: str,
            save_topdir_src: str=           NEMODEL_DEFAULTS['save_topdir'],
            save_topdir_trg: Optional[str]= None,
            save_fn_pfx: str=               NEMODEL_DEFAULTS['save_fn_pfx']):

        if save_topdir_trg is None: save_topdir_trg = save_topdir_src

        # copy DNA with ParaSave
        ParaSave.copy_saved_dna(
            name_src=           name_src,
            name_trg=           name_trg,
            save_topdir_src=    save_topdir_src,
            save_topdir_trg=    save_topdir_trg,
            save_fn_pfx=        save_fn_pfx)

        # copy checkpoints
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

    # performs GX on saved NEModel objects (NEModel as a ParaSave and then checkpoints, without even building child objects)
    @staticmethod
    def gx_saved(
            name_parent_main: str,
            name_parent_scnd: str,
            name_child: str,
            save_topdir_parent_main: str,
            save_topdir_parent_scnd: Optional[str] =    None,
            save_topdir_child: Optional[str] =          None,
            save_fn_pfx: Optional[str] =                NEMODEL_DEFAULTS['save_fn_pfx'],
            do_gx_ckpt=                                 True,
            ratio: float=                               0.5,
            noise: float=                               0.03
    ) -> None:

        # build NEModel and save its ckpt in a separate subprocess
        @proc_wait
        def save(name:str, save_topdir:str):
            nm = NEModel(name=name, save_topdir=save_topdir)
            nm.save_ckpt()

        ParaSave.gx_saved_dna(
            name_parent_main=           name_parent_main,
            name_parent_scnd=           name_parent_scnd,
            name_child=                 name_child,
            save_topdir_parent_main=    save_topdir_parent_main,
            save_topdir_parent_scnd=    save_topdir_parent_scnd,
            save_topdir_child=          save_topdir_child,
            save_fn_pfx=                save_fn_pfx)

        if do_gx_ckpt:
            NEModel.gx_ckpt(
                name_A=         name_parent_main,
                name_B=         name_parent_scnd,
                name_child=     name_child,
                folder_A=       save_topdir_parent_main,
                folder_B=       save_topdir_parent_scnd,
                folder_child=   save_topdir_child,
                ratio=          ratio,
                noise=          noise)
        else: save(
            name=           name_child,
            save_topdir=    save_topdir_child or save_topdir_parent_main)

    # saves NEModel (ParaSave DNA and checkpoint)
    def save(self):
        assert not self['read_only'], f'ERR: cannot save NEModel {self.name} while model is readonly!'
        ParaSave.save_dna(self)
        self.save_ckpt()
        if self.verb>0: print(f'NEMmodel {self.name} saved')

    @property
    def gFWD(self):
        return self._gFWD

    @property
    def session(self):
        return self._session

    @property
    def tbwr(self):
        return self.__TBwr


    def __str__(self):
        # TODO: add some NEModel-specific output (name, graph info, weights)
        return ParaSave.dict_2str(self.get_point())