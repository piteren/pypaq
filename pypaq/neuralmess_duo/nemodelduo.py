# to suppress TF logs from C side by setting an environment variable
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from typing import List, Callable, Dict, Optional
import warnings

from pypaq.lipytools.little_methods import stamp, get_params, get_func_dna
from pypaq.lipytools.logger import set_logger
from pypaq.mpython.devices import get_devices
from pypaq.pms.parasave import ParaSave
from pypaq.neuralmess_duo.base_elements import lr_scaler, grad_clipper_AVT
from pypaq.neuralmess_duo.tbwr import TBwr

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

# defaults below may be given with NEModel_duo kwargs or overridden by fwd_graph attributes
NEMODELDUO_DEFAULTS = {
    'seed':         123,                        # seed for TF and numpy
    'opt_class':    tf.keras.optimizers.Adam,   # default optimizer of train()
    'devices':      'GPU:0',                    # for TF1 we used '/device:CPU:0'
    # LR management (parameters of LR warmup and annealing)
    'iLR':          3e-4,                       # initial learning rate (base)
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
    'hpmser_mode':  False,                      # it will set model to be read_only and quiet
    'read_only':    False,                      # sets model to be read only - wont save anything (wont even create self.model_dir)
    'do_logfile':   True,                       # enables saving log file in self.model_dir
    'do_TB':        True,                       # runs TensorBard, saves in self.model_dir
}


# exemplary FWD function implementation
def fwd_graph(
        in_width=           10,
        hidden_layers=      (128,),
        out_width=          3,
        iLR=                0.0005,     # defaults of fwd_graph will override NEMODELDUO_DEFAULTS
        verb=               0):

    in_vec =  tf.keras.Input(shape=(in_width,), name="in_vec")
    in_true = tf.keras.Input(shape=(1,),        name="in_true")
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

    return {
        'in_vec':   in_vec,
        'in_true':  in_true,
        'probs':    probs,
        'loss':     loss, # loss must be returned to build train_model

        # train_model IO specs, put here inputs and output (keys) of train_model to be built by NEModelDUO
        'train_model_IO':   {
            'inputs':       ['in_vec','in_true'],   # str or List[str]
            'outputs':      'probs'},               # str or List[str], put here any metric tensors, loss may be not added
    }



class NEModelDUO(ParaSave):

    def __init__(
            self,
            name: str,
            fwd_func: Callable,                                 # function building graph (from inputs to loss) - always has to be given
            name_timestamp=             False,                  # adds timestamp to the model name
            save_topdir=                '_models',              # top folder of model save
            save_fn_pfx=                'nemodelduo_dna',       # dna filename prefix
            verb=                       0,
            **kwargs):

        if not tf.executing_eagerly(): warnings.warn(f'TF is NOT executing eagerly!')

        if name_timestamp: name += f'.{stamp()}'

        if verb>0: print(f'\n *** NEModelDUO {name} (type: {type(self).__name__}) *** initializes..')

        # *************************************************************************************************** manage DNA

        dna = {
            'name':        name,
            'save_topdir': save_topdir,
            'save_fn_pfx': save_fn_pfx}

        dna_fwd_func_defaults = get_params(fwd_func)['with_defaults']       # get fwd_func defaults
        dna_saved = ParaSave.load_dna(**dna)                                # load dna from folder

        dna['fwd_func'] = fwd_func
        dna.update(NEMODELDUO_DEFAULTS)
        dna.update(dna_fwd_func_defaults)
        dna.update(dna_saved)
        dna['verb'] = verb
        dna.update(kwargs)                                                  # update with kwargs given NOW by user
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

        self.model_dir = f'{self.save_topdir}/{self.name}'
        if self.verb>0: print(f' > NEModelDUO dir: {self.model_dir}{" read only mode!" if self["read_only"] else ""}')

        if self['do_logfile']:
            set_logger(
                log_folder=     self.model_dir,
                custom_name=    self.name,
                verb=           self.verb)

        dna = self.get_point()

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
        self.device = get_devices(self['devices'], tf2_naming=True, verb=self.verb)[0]

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
        if 'loss' not in self['train_model_IO']['outputs']: self['train_model_IO']['outputs'].append('loss') # add loss

        if self.verb>0: print(f'\n > NEModelDUO is building train_model, name: {self.name} ..')
        self.train_model = self.__get_model(
            name=       self.name,
            inputs=     self['train_model_IO']['inputs'],
            outputs=    self['train_model_IO']['outputs'])

        with tf.device(self.device):

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
            self.train_model.load_weights(filepath=f'{self.model_dir}/weights')
            if self.verb>0: print(f' > train_model weights loaded..')
        except Exception as e:
            if self.verb>0: print(f' > train_model weights NOT loaded ({e})..')

        if self.verb>1:
            print(f'\n >> train.model ({self.train_model.name}) weights:')
            for w in self.train_model.weights: print(f' **  {w.name:30} {str(w.shape):10} {w.device}')

        scaled_LR = lr_scaler(
            iLR=        self['iLR'],
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

        self.writer = TBwr(logdir=self.model_dir, set_to_CPU=False) if self['do_TB'] else None

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
    def __train(self, data):

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
            'iterations':   self['optimizer'].iterations}) # TODO: is iterations saved and kept properly with checkpoint

        return out

    # train wrapped with device
    def train(self, data):
        with tf.device(self.device):
            return self.__train(data)

    def log_TB(self, value, tag: str, step: int):
        if self.writer: self.writer.add(value=value, tag=tag, step=step)
        else: warnings.warn(f'NEModel_duo {self.name} cannot log TensorBoard since do_TB flag is False!')

    # mixes: weights_a * (1-ratio) + weights_a * ratio + noise
    @staticmethod
    def weights_mix(
            weights_a: List[tf.Variable],
            weights_b: List[tf.Variable],
            weights_target: Optional[List[tf.Variable]]=    None,
            ratio: float=                                   0.5,
            noise: float=                                   0.03,
            verb=                                           0):

        if weights_target is None: weights_target = weights_a

        if verb>0: print(f' > weight_mix gots {len(weights_target)} variables to mix')

        for wt, wa, wb in zip(weights_target, weights_a, weights_b):
            assert wt.shape == wa.shape == wb.shape
            assert wt.dtype == wa.dtype == wb.dtype

            if wa.dtype in [tf.float16, tf.float32, tf.float64, tf.double, tf.bfloat16, tf.half]:
                noise_tensor = tf.random.truncated_normal(
                    shape=  wa.shape,
                    stddev= tf.math.reduce_std(wa))

                new_val = wa*(1-ratio) + wb*ratio + noise_tensor*noise
                wt.assign(new_val)
                if verb>0: print(f' >> mixed: {wa.name:50} with {wb.name:50}, {wa.dtype}')
            else:
                wt.assign(wa)
                if verb>0: print(f' >> not mixed: {wa.name:50}, {wa.dtype}')

    def __str__(self):
        return ParaSave.dict_2str(self.get_point())

    def save(self):
        assert not self['read_only'], 'ERR: read only NEModelDUO cannot be saved!'
        self.save_dna()
        self.iterations.assign(self['optimizer'].iterations)
        self.train_model.save_weights(filepath=f'{self.model_dir}/weights')

    def exit(self):
        if self.writer: self.writer.exit()