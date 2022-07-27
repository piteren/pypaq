# to suppress TF logs from C side by setting an environment variable
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from typing import List, Callable, Dict, Optional
import warnings

from pypaq.lipytools.little_methods import stamp, get_params, get_func_dna
from pypaq.mpython.devices import DevicesParam, get_devices
from pypaq.pms.parasave import ParaSave
from pypaq.neuralmess_duo.base_elements import lr_scaler, grad_clipper_AVT
from pypaq.neuralmess_duo.tbwr import TBwr

# restricted keys for fwd_func DNA and return DNA (if they appear in kwargs, should be named exactly like below)
SPEC_KEYS = [
    'train_vars',                                       # list of variables to train (may be returned, otherwise all trainable are taken)
    'opt_vars',                                         # list of variables returned by opt_func
    'acc',                                              # accuracy
    'f1',                                               # F1
    'batch_size',                                       # batch size
    'n_batches']                                        # number of batches for train

# defaults below may be overridden by fwd_graph attributes
NEMODELDUO_DEFAULTS = {
    'seed':             123,                            # seed for TF and numpy
    'opt_class':        tf.keras.optimizers.Adam,       # default optimizer of train()
    'iLR':              3e-4,                           # initial learning rate (base)
    #iLR management (parameters of LR warmup and annealing)
    'warm_up':          None,
    'ann_base':         None,
    'ann_step':         1.0,
    'n_wup_off':        1.0,
    # gradients clipping parameters
    'avt_SVal':         0.1,
    'avt_window':       100,
    'avt_max_upd':      1.5,
    'do_clip':          False,
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
            fwd_func: Callable,                                 # function building graph (from inputs to loss)
            name_timestamp=             False,                  # adds timestamp to the model name
            devices: DevicesParam=      'GPU:0',                # for TF1 we used '/device:CPU:0'
            save_topdir=                '_models',              # top folder of model save
            save_fn_pfx=                'nemodelduo_dna',       # dna filename prefix
            do_TB=                      True,                   # runs TensorBard
            verb=                       0,
            **kwargs):

        if verb>1: tf.debugging.set_log_device_placement(True)
        if name_timestamp: name += f'.{stamp()}'
        if verb>0: print(f'\n *** NEModelDUO {name} (type: {type(self).__name__}) *** initializes..')
        if not tf.executing_eagerly(): warnings.warn(f'TF is NOT executing eagerly!')

        # *************************************************************************** collect DNA from different sources

        dna = {
            'name': name,
            'save_topdir': save_topdir,
            'save_fn_pfx': save_fn_pfx}

        dna_saved = ParaSave.load_dna(**dna)                # load dna from folder

        dna['fwd_func'] = fwd_func                          # update fwd_func
        dna.update(NEMODELDUO_DEFAULTS)                     # update with NEMODELDUO_DEFAULTS
        dna.update(get_params(fwd_func)['with_defaults'])   # update with fwd_func defaults
        dna.update(dna_saved)                               # update with already saved dna
        dna['verb'] = verb                                  # update verb
        dna.update(kwargs)                                  # update with kwargs given NOW by user

        if verb>0:
            print(f'\n > NEModelDUO DNA sources:')
            print(f' >> NEMODELDUO_DEFAULTS:  {NEMODELDUO_DEFAULTS}')
            print(f' >> fwd_func defaults:    {get_params(fwd_func)["with_defaults"]}')
            print(f' >> DNA saved:            {dna_saved}')
            print(f' >> given kwargs:         {kwargs}')

        ParaSave.__init__(self, lock_managed_params=True, **dna)

        self.check_params_sim(SPEC_KEYS)  # safety check

        dna = self.get_point()
        if self.verb>0: print(f'\n > NEModelDUO complete DNA: {dna}')

        # **************************************************************************************************************

        self.dir = f'{self.save_topdir}/{self.name}' # self.save_topdir & self.name come from ParaSave

        np.random.seed(self['seed'])
        tf.random.set_seed(self['seed'])

        # *********************************************************************************************** manage devices

        self.devices = get_devices(devices, tf2_naming=True)[0] # TODO: by now we are using only first device

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
            if self.devices in dev.name or 'CPU' in dev.name:
                set_visible.append(dev)
        if self.verb>1: print(f' > setting visible to NEModelDUO: {set_visible} for {self.devices}')
        tf.config.set_visible_devices(set_visible)

        # **************************************************************************************************************

        if self.verb>0: print(f'\n > building graph ({fwd_func}) on {self.devices} ..')
        fwd_func_dna = get_func_dna(fwd_func, dna)
        with tf.device(self.devices):
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

        with tf.device(self.devices):

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
            self.train_model.load_weights(filepath=f'{self.dir}/weights')
            if self.verb>0: print(f' > train_model weights loaded..')
        except:
            if self.verb>0: print(f' > train_model weights NOT loaded..')

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

        self.writer = TBwr(logdir=self.dir, set_to_CPU=False) if do_TB else None

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
    def __call(
            self,
            data,
            name: Optional[str] = None,
            training=False):

        if name is None:
            model = self.train_model
        else:
            assert name in self.submodels
            model = self.submodels[name]

        if self.verb > 1: print(
            f' >> NEModelDUO is calling: {model.name}, inputs: {model.inputs}, outputs: {model.outputs}')
        return model(data, training=training)

    # call wrapped with device
    def call(self, **kwargs):
        with tf.device(self.devices):
            return self.__call(**kwargs)

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
    def train(self, **kwargs):
        with tf.device(self.devices):
            return self.__train(**kwargs)

    def log_TB(self, value, tag: str, step: int):
        if self.writer: self.writer.add(value=value, tag=tag, step=step)
        else: warnings.warn(f'NEModel_duo {self.name} cannot log TensorBoard since do_TB flag is False!')

    def save(self):
        self.save_dna()
        self.iterations.assign(self['optimizer'].iterations)
        self.train_model.save_weights(filepath=f'{self.dir}/weights')

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

        if verb > 0: print(f' > weight_mix gots {len(weights_target)} variables to mix')

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

    def exit(self):
        if self.writer: self.writer.exit()