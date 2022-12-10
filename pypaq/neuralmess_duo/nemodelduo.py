# to suppress TF logs from C side by setting an environment variable
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from typing import List, Callable, Dict, Optional
import warnings

from pypaq.lipytools.little_methods import stamp, get_params, get_func_dna
from pypaq.lipytools.logger import set_logger
from pypaq.lipytools.moving_average import MovAvg
from pypaq.mpython.devices import get_devices
from pypaq.pms.parasave import ParaSave
from pypaq.neuralmess_duo.base_elements import lr_scaler, grad_clipper_AVT
from pypaq.neuralmess_duo.tbwr import TBwr
from pypaq.comoneural.batcher import Batcher

# restricted keys for fwd_func DNA and return DNA (if they appear in kwargs, should be named exactly like below)
SPEC_KEYS = [
    'name',         # model name
    'train_vars',   # list of variables to train (may be returned, otherwise all trainable are taken)
    'opt_vars',     # list of variables returned by opt_func
    'loss',         # loss
    'acc',          # accuracy
    'f1',           # F1
    'batch_size',   # batch size
    'n_batches',    # number of batches for train
    'verb',         # verbosity
]

# defaults below may be given with NEModelDUO kwargs or overridden by fwd_graph attributes
NEMODELDUO_DEFAULTS = {
    'seed':         123,                        # seed for TF and numpy
    'opt_class':    tf.keras.optimizers.Adam,   # default optimizer of train()
    'devices':      'GPU:0',                    # for TF1 we used '/device:CPU:0'
    # LR management (parameters of LR warmup and annealing)
    'baseLR':          3e-4,                       # initial learning rate (base init)
    'warm_up':      None,
    'ann_base':     None,
    'ann_step':     1.0,
    'n_wup_off':    1.0,
    # gradients clipping parameters
    'avt_SVal':     0.1,
    'avt_window':   100,
    'avt_max_upd':  1.5,
    'do_clip':      False,
    # other
    'save_topdir':  '_models',                  # top folder of model save
    'save_fn_pfx':  'nemodelduo_dna',           # dna filename prefix
    'hpmser_mode':  False,                      # it will set model to be read_only and quiet when running with hpmser
    'read_only':    False,                      # sets model to be read only - wont save anything (wont even create self.nnwrap_dir)
    'do_logfile':   True,                       # enables saving log file in self.nnwrap_dir
    'do_TB':        True}                       # runs TensorBard, saves in self.nnwrap_dir


# exemplary FWD function implementation
def fwd_graph(
        in_width=           10,
        hidden_layers=      (128,),
        out_width=          3,
        baseLR=                0.0005,     # defaults of fwd_graph will override NEMODELDUO_DEFAULTS
        verb=               0):

    in_vec =  tf.keras.Input(shape=in_width, name="in_vec")
    in_true = tf.keras.Input(shape=1,        name="in_true", dtype=tf.int32)
    if verb>0: print(' > in_vec', in_vec)

    lay = in_vec
    for i in range(len(hidden_layers)):
        lay = tf.keras.layers.Dense(
            name=       f'hidden_layer_{i}',
            units=      hidden_layers[i],
            activation= tf.nn.relu)(lay)

    logits = tf.keras.layers.Dense(
        name=       'value',
        units=      out_width,
        activation= None)(lay)
    if verb>0: print(' > logits', logits)

    probs = tf.nn.softmax(logits)

    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true=         in_true,
        y_pred=         logits,
        from_logits=    True)
    loss = tf.reduce_mean(loss)
    if verb > 0: print(' > loss', loss)

    preds = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(in_true), preds), dtype=tf.float32))

    return {
        'in_vec':   in_vec,
        'in_true':  in_true,
        'probs':    probs,
        'loss':     loss, # loss must be returned to build train_model
        'acc':      acc,

        # train_model IO specs, put here inputs and output (keys) of train_model to be built by NEModelDUO
        'train_model_IO':   {
            'inputs':       ['in_vec', 'in_true'],  # str or List[str]
            'outputs':      ['probs', 'acc']},      # str or List[str], put here any metric tensors, loss may be not added
    }



class NEModelDUO(ParaSave):

    def __init__(
            self,
            name: str,
            fwd_func: Optional[Callable]=   None,       # function building graph (from inputs to loss) - may be not given if model already saved
            name_timestamp=                 False,      # adds timestamp to the model name
            save_topdir=                    NEMODELDUO_DEFAULTS['save_topdir'],
            save_fn_pfx=                    NEMODELDUO_DEFAULTS['save_fn_pfx'],
            verb=                           0,
            **kwargs):

        if not tf.executing_eagerly(): warnings.warn(f'TF is NOT executing eagerly!')

        if name_timestamp: name += f'.{stamp()}'
        if verb>0: print(f'\n *** NEModelDUO {name} (type: {type(self).__name__}) *** initializes..')

        # ***************************************************************************************** manage (resolve) DNA

        # load dna from folder
        dna_saved = ParaSave.load_dna(
            name=           name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx)

        if not fwd_func:
            assert 'fwd_func' in dna_saved, 'ERR: cannot continue: fwd_func was not given and model ahs not been saved before, please give fwd_func to model init!'
            fwd_func = dna_saved['fwd_func']

        dna_fwd_func_defaults = get_params(fwd_func)['with_defaults']  # get fwd_func defaults

        dna = {}
        dna.update(NEMODELDUO_DEFAULTS)
        dna.update(dna_fwd_func_defaults)
        dna.update(dna_saved)
        dna.update(kwargs)                  # update with kwargs given NOW by user
        dna.update({
            'name':         name,
            'save_topdir':  save_topdir,
            'save_fn_pfx':  save_fn_pfx,
            'fwd_func':     fwd_func,
            'verb':         verb})

        ParaSave.__init__(self, lock_managed_params=True, **dna)
        self.check_params_sim(SPEC_KEYS + list(NEMODELDUO_DEFAULTS.keys())) # safety check

        # hpmser_mode - early override
        if self['hpmser_mode']:
            self.verb = 0
            self['read_only'] = True

        # read only - early override
        if self['read_only']:
            self['do_logfile'] = False
            self['do_TB'] = False

        self.nnwrap_dir = f'{self.save_topdir}/{self.name}'
        if self.verb>0: print(f' > NEModelDUO dir: {self.nnwrap_dir}{" read only mode!" if self["read_only"] else ""}')

        if self['do_logfile']:
            set_logger(
                log_folder=     self.nnwrap_dir,
                custom_name=    self.name,
                verb=           self.verb)

        dna = self.get_point()  # bake

        if self.verb>0:
            print(f'\n > NEModelDUO DNA sources:')
            print(f' >> NEMODELDUO_DEFAULTS:    {NEMODELDUO_DEFAULTS}')
            print(f' >> fwd_func defaults:      {dna_fwd_func_defaults}')
            print(f' >> DNA saved:              {dna_saved}')
            print(f' >> given kwargs:           {kwargs}')
            print(f' NEModelDUO complete DNA:   {dna}')

        # **************************************************************************************************************

        np.random.seed(self['seed'])
        tf.random.set_seed(self['seed'])

        # *********************************************************************************************** manage devices

        # TODO: by now we are using only first device, support for multidevice will be added later
        self.device = get_devices(self['devices'], namespace='TF2')[0]
        if self.verb>1: print(f' > setting devices from {self["devices"]} to {self.device}')

        # check for TF visible_physical_devices
        visible_physical_devices = tf.config.list_physical_devices()
        if self.verb>1:
            print(' > devices available for TF:')
            for dev in visible_physical_devices:
                print(f' >> {dev.name}')

        # INFO: Currently, memory growth needs to be the same across GPUs (https://www.tensorflow.org/guide/gpu#logging_device_placement)
        for dev in visible_physical_devices:
            if 'GPU' in dev.name:
                tf.config.experimental.set_memory_growth(dev, True)

        set_visible = []
        for dev in visible_physical_devices:
            # INFO: it looks that TF usually needs CPU and it cannot be masked-out
            # TODO: self.device in dev.name will fail for more than 10 GPUs, cause 'GPU:1' is in 'GPU:10' - need to fix it later
            if self.device in dev.name or 'CPU' in dev.name:
                set_visible.append(dev)
        if self.verb>1: print(f' > setting visible to NEModelDUO: {set_visible} for {self.device}')
        tf.config.set_visible_devices(set_visible)

        # **************************************************************************************************************

        if self.verb>0: print(f'\n > building graph ({fwd_func}) on {self.device} ..')
        if self.verb>1: tf.debugging.set_log_device_placement(True)

        fwd_func_dna = get_func_dna(fwd_func, dna)
        with tf.device(self.device):
            fwd_func_out = fwd_func(**fwd_func_dna)
        self.update(fwd_func_out)

        assert 'loss' in self, 'ERR: You need to return loss with fwd_func!'
        assert 'train_model_IO' in self, 'ERR: fwd_func should return train_model_IO specs, see fwd_graph example!'

        # change to lists
        for k in self['train_model_IO']:
            if type(self['train_model_IO'][k]) is str:
                self['train_model_IO'][k] = [self['train_model_IO'][k]]
        if 'outputs' not in self['train_model_IO']: self['train_model_IO']['outputs'] = []
        if 'loss' not in self['train_model_IO']['outputs']: self['train_model_IO']['outputs'].append('loss') # add loss

        if self.verb>0: print(f'\n > NEModelDUO is building train_model, name: {self.name} ..')
        self.train_model = self.__get_model(
            name=       self.name,
            inputs=     self['train_model_IO']['inputs'],
            outputs=    self['train_model_IO']['outputs'])

        with tf.device(self.device):

            # variable for time averaged global norm of gradients
            self.lr = self.train_model.add_weight(
                name=       'lr',
                trainable=  False,
                dtype=      tf.float32)
            self.lr.assign(self['baseLR'])

            # variable for time averaged global norm of gradients
            self.ggnorm_avt = self.train_model.add_weight(
                name=       'gg_avt_norm',
                trainable=  False,
                dtype=      tf.float32)
            self.ggnorm_avt.assign(self['avt_SVal'])

            # global step variable
            self.iterations = self.train_model.add_weight(
                name=       'iterations',
                trainable=  False,
                dtype=      tf.int64)
            self.iterations.assign(0)

        try:
            self.train_model.load_weights(filepath=f'{self.nnwrap_dir}/weights')
            if self.verb>0: print(f' > train_model weights loaded..')
        except Exception as e:
            if self.verb>0: print(f' > train_model weights NOT loaded ({e})..')

        if self.verb>1:
            print(f'\n >> train.model ({self.train_model.name}) weights:')
            for w in self.train_model.weights: print(f' **  {w.name:30} {str(w.shape):10} {w.device}')

        scaled_LR = lr_scaler(
            baseLR=        self.lr,
            g_step=     self.iterations,
            warm_up=    self['warm_up'],
            ann_base=   self['ann_base'],
            ann_step=   self['ann_step'],
            n_wup_off=  self['n_wup_off'],
            verb=       self.verb)
        if self.verb>0: print('scaled_LR', scaled_LR)
        self.optimizer = self['opt_class'](learning_rate=scaled_LR)
        self.optimizer.iterations = self.iterations

        self.submodels: Dict[str, tf.keras.Model] = {}

        self.writer = TBwr(logdir=self.nnwrap_dir, set_to_CPU=False) if self['do_TB'] else None

        self._model_data = None
        self._batcher = None

        if self.verb>0: print(f'\n > NEModelDUO init finished..')


    def __get_model(
            self,
            name: str,
            inputs: str or List[str],
            outputs: str or List[str]) -> tf.keras.Model:
        if type(inputs) is str:  inputs = [inputs]
        if type(outputs) is str: outputs = [outputs]
        for n in inputs + outputs: assert n in self, f'ERR: {n} not in self!'
        return tf.keras.Model(
            name=       name,
            inputs=     {n: self[n] for n in inputs},
            outputs=    {n: self[n] for n in outputs})


    def build_callable_submodel(
            self,
            name: str,
            inputs: str or List[str],
            outputs: str or List[str]):
        assert name not in self.submodels
        if self.verb>0: print(f'\n > NEModelDUO is building callable: {name}, inputs: {inputs}, outputs: {outputs}')
        self.submodels[name] = self.__get_model(
            name=       name,
            inputs=     inputs,
            outputs=    outputs)

    # call wrapped with tf.function
    @tf.function(reduce_retracing=True)
    def __call(self, data, name):

        if name is None: model = self.train_model
        else:
            assert name in self.submodels
            model = self.submodels[name]

        if self.verb > 1: print(
            f' >> NEModelDUO is calling: {model.name}, inputs: {model.inputs}, outputs: {model.outputs}')
        return model(data, training=False)

    # call wrapped with device
    def call(self, data, name: Optional[str]=None):
        with tf.device(self.device):
            return self.__call(data, name)

    # train wrapped with tf.function
    @tf.function(reduce_retracing=True)
    def __train_batch(self, data):

        with tf.GradientTape() as tape:
            out = self.train_model(data, training=True)

        # TODO: what about colocate_gradients_with_ops=False
        gradients = tape.gradient(
            target=     out['loss'],
            sources=    self.train_model.trainable_variables)

        gclr_out = grad_clipper_AVT(
            variables=      self.train_model.trainable_variables,
            gradients=      gradients,
            ggnorm_avt=     self.ggnorm_avt,
            optimizer=      self['optimizer'],
            avt_window=     self['avt_window'],
            avt_max_upd=    self['avt_max_upd'],
            do_clip=        self['do_clip'],
            verb=           self.verb)

        out.update({
            'ggnorm':       gclr_out['ggnorm'],
            'ggnorm_avt':   self.ggnorm_avt,
            'iterations':   self['optimizer'].iterations})

        return out

    # __train_batch wrapped with device
    def train_batch(self, data):
        with tf.device(self.device):
            return self.__train_batch(data)

    # ******************************************************************** baseline methods for training with given data

    # loads model data for training, dict should have at least 'train':{} for Batcher
    def load_model_data(self) -> dict:
        warnings.warn('NEModelBase.load_model_data() should be overridden!')
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

    # training method, saves for max_ts_acc
    # INFO: training method below is based on model accuracy so it should be returned by graph as 'acc'
    def train(
            self,
            test_freq=      100,  # number of batches between tests, model SHOULD BE tested while training
            mov_avg_factor= 0.1,
            save=           True):  # allows to save model while training

        self.pre_train()

        if self.verb>0: print(f'{self.name} - training starts')
        batch_IX = 0
        tr_lssL = []
        tr_accL = []
        ts_acc_max = 0
        ts_acc_mav = MovAvg(mov_avg_factor)

        ts_results = []
        ts_bIX = [bIX for bIX in range(self['n_batches']+1) if not bIX % test_freq] # batch indexes when test will be performed
        assert ts_bIX, 'ERR: model SHOULD BE tested while training!'
        ten_factor = int(0.1*len(ts_bIX)) # number of tests for last 10% of training
        if ten_factor < 1: ten_factor = 1 # we need at least one result
        if self['hpmser_mode']: ts_bIX = ts_bIX[-ten_factor:]

        while batch_IX < self['n_batches']:
            batch_IX += 1
            batch = self._batcher.get_batch()

            run_out = self.train_batch(batch)

            if self['do_TB']:
                for k in ['loss', 'acc', 'ggnorm', 'ggnorm_avt']:
                    self.log_TB(value=run_out[k], tag=f'tr/{k}', step=batch_IX)

            if self.verb>0:
                tr_lssL.append(run_out['loss'])
                tr_accL.append(run_out['acc'])

            if batch_IX in ts_bIX:
                ts_out = self.test()
                ts_results.append(ts_out['acc'])
                ts_out['acc_mav'] = ts_acc_mav.upd(ts_out['acc'])
                if self['do_TB']:
                    for k in ['loss', 'acc', 'acc_mav']:
                        self.log_TB(value=ts_out[k], tag=f'ts/{k}', step=batch_IX)
                if self.verb>0: print(f'{batch_IX:5d} TR: {100*sum(tr_accL)/test_freq:.1f} / {sum(tr_lssL)/test_freq:.3f} -- TS: {100*ts_out["acc"]:.1f} / {ts_out["loss"]:.3f}')
                tr_lssL = []
                tr_accL = []

                if ts_out['acc'] > ts_acc_max:
                    ts_acc_max = ts_out['acc']
                    if not self['read_only'] and save: self.save()  # model is saved for max_ts_acc

        # weighted test value for last 10% test results
        ts_results = ts_results[-ten_factor:]
        ts_wval = 0
        weight = 1
        sum_weight = 0
        for tr in ts_results:
            ts_wval += tr * weight
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
        lossL = []
        accL = []
        for batch in batches:
            out = self.train_model(batch, training=False)
            lossL.append(out['loss'])
            accL.append(out['acc'])
        return {
            'acc':  sum(accL)/len(accL),
            'loss': sum(lossL)/len(lossL)}


    def log_TB(self, value, tag: str, step: int):
        if self.writer: self.writer.add(value=value, tag=tag, step=step)
        else: warnings.warn(f'NEModelDUO {self.name} cannot log TensorBoard since do_TB flag is False!')

    # updates base LR (baseLR) in graph - but not saves it to the checkpoint
    def update_LR(self, lr: float):
        old = self['baseLR']
        self['baseLR'] = lr
        self.lr.assign(self['baseLR'])
        if self.verb>1: print(f'NEModelDUO {self.name} updated baseLR from {old} to {self["baseLR"]}')


    def __str__(self):
        s = f'NEModelDUO {self.name}'
        s += f'\n > baseLR: {self.lr.numpy()}'
        s += f'\n > iterations: {self.iterations.numpy()}'
        s += f'\ntrain_model weights:'
        total_w = 0
        for w in self.train_model.weights:
            s += f'\n > {w.name:30} {str(w.shape):10} {w.device}'
            shape = w.shape
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            total_w += variable_parameters
        s += f'\n ### total variables: {total_w}'
        s += f'\nModel params:\n{ParaSave.dict_2str(self.get_point())}'
        return s


    def save(self):
        assert not self['read_only'], 'ERR: read only NEModelDUO cannot be saved!'
        self.save_dna()
        self.iterations.assign(self['optimizer'].iterations)
        self.train_model.save_weights(filepath=f'{self.nnwrap_dir}/weights')
        if self.verb>0: print(f'model {self.name} saved')


    def exit(self):
        if self.writer: self.writer.exit()

    # copies full NEModelDUO folder (DNA & checkpoints)
    @staticmethod
    def copy_saved(
            name_src: str,
            name_trg: str,
            save_topdir_src: str=           NEMODELDUO_DEFAULTS['save_topdir'],
            save_topdir_trg: Optional[str]= None,
            save_fn_pfx: str=               NEMODELDUO_DEFAULTS['save_fn_pfx']):

        if save_topdir_trg is None: save_topdir_trg = save_topdir_src

        # copy DNA with ParaSave
        ParaSave.copy_saved_dna(
            name_src=           name_src,
            name_trg=           name_trg,
            save_topdir_src=    save_topdir_src,
            save_topdir_trg=    save_topdir_trg,
            save_fn_pfx=        save_fn_pfx)

        # load dna from folder & build child model
        dna_trg = ParaSave.load_dna(
            name=           name_trg,
            save_topdir=    save_topdir_trg,
            save_fn_pfx=    save_fn_pfx)
        model_trg = NEModelDUO(**dna_trg)

        ckptA_FD = f'{save_topdir_src}/{name_src}/'
        model_trg.train_model.load_weights(filepath=f'{ckptA_FD}/weights')

        ckptC_FD = f'{save_topdir_trg}/{name_trg}/'
        model_trg.train_model.save_weights(filepath=f'{ckptC_FD}/weights')

        model_trg.exit()

    # returns train_model weights (returns list of numpy.ndarray, numpy.float32, numpy.int64, ..)
    def get_weights(self) -> list:
        return [w.numpy() for w in self.train_model.weights]

    @staticmethod
    def gx_saved(
            name_parent_main: str,
            name_parent_scnd: Optional[str],            # if not given makes GX only with main parent
            name_child: str,
            save_topdir_parent_main: str=               NEMODELDUO_DEFAULTS['save_topdir'],
            save_topdir_parent_scnd: Optional[str] =    None,
            save_topdir_child: Optional[str] =          None,
            save_fn_pfx: Optional[str] =                NEMODELDUO_DEFAULTS['save_fn_pfx'],
            ratio: float=                               0.5,
            noise: float=                               0.03) -> None:

        if not save_topdir_parent_scnd: save_topdir_parent_scnd = save_topdir_parent_main
        if not save_topdir_child: save_topdir_child = save_topdir_parent_main

        # GX ParaSave dna
        ParaSave.gx_saved_dna(
            name_parent_main=           name_parent_main,
            name_parent_scnd=           name_parent_scnd,
            name_child=                 name_child,
            save_topdir_parent_main=    save_topdir_parent_main,
            save_topdir_parent_scnd=    save_topdir_parent_scnd,
            save_topdir_child=          save_topdir_child,
            save_fn_pfx=                save_fn_pfx)

        # load dna from folder & build child model
        dna_saved = ParaSave.load_dna(
            name=           name_child,
            save_topdir=    save_topdir_child,
            save_fn_pfx=    save_fn_pfx)
        model_child = NEModelDUO(**dna_saved)

        ckptA_FD = f'{save_topdir_parent_main}/{name_parent_main}/'
        model_child.train_model.load_weights(filepath=f'{ckptA_FD}/weights')
        weights_A = model_child.get_weights()

        if name_parent_scnd:
            ckptB_FD = f'{save_topdir_parent_scnd}/{name_parent_scnd}/'
            model_child.train_model.load_weights(filepath=f'{ckptB_FD}/weights')

        for w,a in zip(model_child.train_model.weights, weights_A):
            if np.issubdtype(a.dtype, np.floating):
                noise_tensor = tf.random.truncated_normal(
                    shape=  w.shape,
                    stddev= tf.math.reduce_std(w))
                new_val = w * (1 - ratio) + a * ratio + noise_tensor * noise
                w.assign(new_val)

        ckptC_FD = f'{save_topdir_child}/{name_child}/'
        model_child.train_model.save_weights(filepath=f'{ckptC_FD}/weights')

        model_child.exit()

        """
        # other unsuccessful tryouts to load wieghts from TF2 checkpoint:
        
        ckpt = tf.train.load_checkpoint(ckptA_FD)
        print(ckpt)
        print(tf.train.list_variables(ckptA_FD))
        
        ckpt = tf.train.Checkpoint()
        print(ckpt)
        ckpt.restore(tf.train.latest_checkpoint(ckptA_FD))
        print(ckpt)
        
        lsv = tf.train.list_variables(tf.train.latest_checkpoint(ckptA_FD))
        print(lsv)
        for e in lsv:
            v = tf.train.load_variable(ckptA_FD, e[0])
            print(type(v))
        """