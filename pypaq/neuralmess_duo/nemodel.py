# to suppress TF logs from C side by setting an environment variable
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from typing import List

from pypaq.pms.subscriptable import Subscriptable
from pypaq.neuralmess_duo.base_elements import grad_clipper_AVT


class NEModel(Subscriptable):

    def __init__(
            self,
            graph,
            optimizer=  tf.keras.optimizers.Adam):

        self.update(graph())
        # TODO: update self with defaults of graph
        # TODO: optimizer with iLR will be managed by agn function
        if 'optimizer' not in self: self['optimizer'] = optimizer
        self['optimizer'] = self['optimizer']()

        if 'loss' in self:
            for k in self['train_model_IO']: self['train_model_IO'][k] = list(self['train_model_IO'][k])            # change to lists
            if 'loss' not in self['train_model_IO']['outputs']: self['train_model_IO']['outputs'].append('loss')    # add loss
            self.build_callable(
                name=       'train_model',
                inputs=     self['train_model_IO']['inputs'],
                outputs=    self['train_model_IO']['outputs'])

            # variable for time averaged global norm of gradients
            # TODO: is this variable properly kept by Model??
            self.gg_avt_norm = self['train_model'].add_weight(
                name=       'gg_avt_norm',
                trainable=  False,
                dtype=  tf.float32)
            avt_SVal = 0.1  # start value for AVT (smaller value makes warmup)
            self.gg_avt_norm.assign(avt_SVal)

        self.writer = tf.summary.create_file_writer('_models/new')

    def build_callable(
            self,
            name: str,
            inputs: str or List[str],
            outputs: str or List[str]):
        if type(inputs) is str:  inputs = [inputs]
        if type(outputs) is str: outputs = [outputs]
        self[name] = tf.keras.Model(
            name=       name,
            inputs=     {n: self[n] for n in inputs},
            outputs=    {n: self[n] for n in outputs})

    @tf.function
    def call(self, name: str, data):
        # name should be in self and should be type of tf.keras.engine.functional.Functional, but we will not check it
        print(type(self[name]))
        print(self[name].inputs)
        print(self[name].inputs[0].name)
        print(self[name].outputs)
        print(self[name].outputs[0].name)
        return self[name](data)

    @tf.function
    def train(self, data):

        assert 'train_model' in self, 'ERR: cannot train since train_model has not been defined!'

        with tf.GradientTape() as tape:
            out = self['train_model'](data, training=True)
            print(list(out.keys()))
            print(out['loss'])
        # TODO: losses values will be returned as a metric by NEModel from gradient tape, similar with gn, gn_avt
        variables = self['train_model'].trainable_variables
        # TODO: what about colocate_gradients_with_ops=False
        gradients = tape.gradient(
            target=     out['loss'],
            sources=    variables)

        gclr_out = grad_clipper_AVT(
            variables=      variables,
            gradients=      gradients,
            gg_avt_norm=    self.gg_avt_norm,
            optimizer=      self['optimizer'])
        print(gclr_out)

        # TODO: is iterations saved and kept properly with checkpoint
        print(self['optimizer'].iterations)

        with self.writer.as_default():
            tf.summary.write('loss', out['loss'], step=self['optimizer'].iterations)

        self.writer.flush()