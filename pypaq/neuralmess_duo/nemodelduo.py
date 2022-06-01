# to suppress TF logs from C side by setting an environment variable
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from typing import List, Callable, Dict, Optional

from pypaq.lipytools.little_methods import short_scin, stamp, get_params
from pypaq.pms.parasave import ParaSave
from pypaq.neuralmess_duo.base_elements import grad_clipper_AVT

# restricted keys for fwd_func DNA and return DNA (if they appear in kwargs, should be named exactly like below)
SPEC_KEYS = [
    'name',                                             # model name
    'seed',                                             # seed for TF and numpy
    'iLR',                                              # initial learning rate (base)
    'warm_up','ann_base','ann_step','n_wup_off',        # LR management (parameters of LR warmup and annealing)
    'avt_SVal','avt_window','avt_max_upd','do_clip',    # gradients clipping parameters
    'train_vars',                                       # list of variables to train (may be returned, otherwise all trainable are taken)
    'opt_vars',                                         # list of variables returned by opt_func
    'loss',                                             # loss
    'acc',                                              # accuracy
    'f1',                                               # F1
    'batch_size',                                       # batch size
    'n_batches']                                        # number of batches for train

NEMODELDUO_DEFAULTS = {
    'save_topdir':      '_models',                      # top folder of model save
    'save_fn_pfx':      'nemodelduo_dna',
    'opt_class':        tf.keras.optimizers.Adam,       # default optimizer of train()
}


# exemplary FWD function implementation
def fwd_graph(
        in_width=           10,
        hidden_layers=      (128,),
        out_width=          3,
        iLR=                0.0003,
        verb=               0):

    in_vec =  tf.keras.Input(shape=(in_width,), name="in_vec")
    in_true = tf.keras.Input(shape=(1,),        name="in_true")
    if verb>0: print('in_vec', in_vec)

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
    if verb>0: print('logits', logits)

    probs = tf.nn.softmax(logits)

    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true=         in_true,
        y_pred=         logits,
        from_logits=    True)
    loss = tf.reduce_mean(loss)
    if verb > 0: print('loss', loss)

    return {
        'in_vec':   in_vec,
        'in_true':  in_true,
        'probs':    probs,
        'loss':     loss, # loss must be defined to build train model

        # train_model IO specs
        'train_model_IO':   {
            'inputs':       ['in_vec','in_true'],   # str or List[str]
            'outputs':      'probs'},               # str or List[str], put here any metric tensors, loss may be not added
    }



class NEModelDUO(ParaSave):

    def __init__(
            self,
            name: str,
            name_timestamp=             False,                  # adds timestamp to model name
            fwd_func: Callable=         fwd_graph,              # function building forward (FWD) graph (from inputs to loss)
            save_topdir=                '_models',              # top folder of model save
            save_fn_pfx=                'nemodelduo_dna',
            verb=                       0,
            **kwargs):

        self.verb = verb
        self.name = name
        if name_timestamp: self.name += f'.{stamp()}'
        if self.verb>0: print(f'\n *** NEModelDUO {self.name} (type: {type(self).__name__}) *** initializes...')

        self.save_topdir = save_topdir
        self.save_fn_pfx = save_fn_pfx

        # ******************************************************* collect DNA from different sources and build final DNA

        dna_def = NEMODELDUO_DEFAULTS
        self.__managed_params = self.get_all_fields()  # save managed params (temporary)
        dna_self = self.get_point()
        dna_saved = ParaSave.load_dna(name=self.name, save_topdir=self.save_topdir, save_fn_pfx=self.save_fn_pfx)

        self.fwd_func = dna_saved.get('fwd_func', fwd_func) # get fwd_func if saved

        dna_fwd_func = get_params(self.fwd_func)['with_defaults']

        if self.verb>0:
            print(f'\n > NEModelDUO DNA sources:')
            print(f' >> NEModelDUO defaults:    {dna_def}')
            print(f' >> FWD func defaults:      {dna_fwd_func}')
            print(f' >> DNA saved:              {dna_saved}')
            print(f' >> NEModel DNA:            {dna_self}')
            print(f' >> given kwargs:           {kwargs}')

        self.update(dna_def)        # update with NEModelBase defaults
        self.update(dna_fwd_func)   # update with FWD func defaults
        self.update(dna_saved)      # update with saved DNA
        self.update(dna_self)       # update with early self DNA (should not be updated by any of above)
        self.update(kwargs)         # update with given kwargs

        self.__managed_params = self.get_all_fields()   # save managed params here, graph will add many params that we do not want to be managed

        # TODO: it should check (already in self + SPEC)^2
        self.check_params_sim(SPEC_KEYS)                # safety check

        dna = self.get_point()
        if self.verb>0: print(f'\n > NEModel complete DNA: {dna}')

        ParaSave.__init__(self, **dna)


        self.dir = f'{self.save_topdir}/{self.name}'


        # TODO: do we load weights or load whole model? Do we need build graph then - probably yes -> then just load weights
        # TODO: put here fwd_func params from self
        self.update(fwd_func())

        # TODO: update self with defaults of graph
        # TODO: optimizer with iLR will be managed by agn function
        self.optimizer = self['opt_class']()

        assert 'loss' in self, 'ERR: You need to return loss with fwd_func!'
        assert 'train_model_IO' in self, 'ERR: You need to return train_model_IO specs with fwd_func!'

        # change to lists
        for k in self['train_model_IO']:
            if type(self['train_model_IO'][k]) is str:
                self['train_model_IO'][k] = [self['train_model_IO'][k]]
        if 'loss' not in self['train_model_IO']['outputs']: self['train_model_IO']['outputs'].append('loss') # add loss

        self.train_model = self.__get_model(
            name=       name,
            inputs=     self['train_model_IO']['inputs'],
            outputs=    self['train_model_IO']['outputs'])

        # variable for time averaged global norm of gradients
        # TODO: is this variable properly kept by Model while saving??
        self.ggnorm_avt = self.train_model.add_weight(
            name=       'gg_avt_norm',
            trainable=  False,
            dtype=      tf.float32)
        avt_SVal = 0.1  # start value for AVT (smaller value makes warmup)
        self.ggnorm_avt.assign(avt_SVal)

        self.iterations = self.train_model.add_weight(
            name=       'iterations',
            trainable=  False,
            dtype=      tf.int64)

        self.train_model = tf.keras.models.load_model(filepath=self.dir)

        print(self.train_model.get_weights()[-2:])
        self.optimizer.iterations = self.iterations

        self.submodels: Dict[str, tf.keras.Model] = {}

        self.writer = tf.summary.create_file_writer(self.dir)


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
        if self.verb>0: print(f' > building callable: {name}, inputs: {inputs}, outputs: {outputs}')
        self.submodels[name] = self.__get_model(
            name=       name,
            inputs=     inputs,
            outputs=    outputs)

    @tf.function
    def call(
            self,
            data,
            name: Optional[str]=    None,
            training=               False):

        if name is None: model = self.train_model
        else:
            assert name in self.submodels
            model = self.submodels[name]

        print(type(model))
        print(model.inputs)
        print(model.inputs[0].name)
        print(model.outputs)
        print(model.outputs[0].name)
        return model(data, training=training)

    @tf.function
    def train(self, data):

        with tf.GradientTape() as tape:
            out = self.call(data, training=True)
            loss = out['loss']
            print(list(out.keys()))
            print(loss)

        variables = self.train_model.trainable_variables
        # TODO: what about colocate_gradients_with_ops=False
        gradients = tape.gradient(
            target=     loss,
            sources=    variables)

        gclr_out = grad_clipper_AVT(
            variables=  variables,
            gradients=  gradients,
            ggnorm_avt= self.ggnorm_avt,
            optimizer=  self['optimizer'])

        with self.writer.as_default():
            tf.summary.write('loss', out['loss'], step=self['optimizer'].iterations)

        self.writer.flush()

        return {
            'loss':         loss,
            'ggnorm':       gclr_out['ggnorm'],
            'ggnorm_avt':   self.ggnorm_avt,
            'iterations':   self['optimizer'].iterations} # TODO: is iterations saved and kept properly with checkpoint

    def save(self):
        self.iterations.assign(self['optimizer'].iterations)
        tf.keras.models.save_model(
            model=      self.train_model,
            filepath=   self.dir)

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