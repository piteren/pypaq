from typing import List

from pypaq.neuralmess.base_elements import tf
from pypaq.neuralmess.layers import lay_dense, zeroes


def pga_graph(
        name=               'pg_actor_graph',
        observation_width=  4,
        num_actions=        2,
        hidden_layers=      (20,),
        lay_norm=           False,
        seed=               121):

    with tf.variable_scope(name):

        observation_PH = tf.placeholder(
            shape=  (None, observation_width),
            dtype=  tf.float32,
            name=   'observation')
        action_PH = tf.placeholder(
            shape=  None,
            dtype=  tf.int32,
            name=   'action')
        return_PH = tf.placeholder( # discounted accumulated return
            shape=  None,
            dtype=  tf.float32,
            name=   'return')

        zsL: List[tf.Tensor] = []  # list of zeroes Tensors

        layer = observation_PH
        if lay_norm: layer = tf.keras.layers.LayerNormalization(axis=-1)(layer)
        for i in range(len(hidden_layers)):
            layer = lay_dense(
                input=      layer,
                name=       f'hidden_layer_{i}',
                units=      hidden_layers[i],
                activation= tf.nn.relu,
                seed=       seed)
            zsL.append(zeroes(layer))
            if lay_norm: layer = tf.keras.layers.LayerNormalization(axis=-1)(layer)

        action_logits = lay_dense(
            input=      layer,
            name=       'action_logits',
            units=      num_actions,
            activation= None,
            seed=       seed)
        probs = tf.nn.softmax(action_logits)

        # actor_ce = tf.losses.sparse_softmax_cross_entropy(logits=action_logits, labels=action_PH, weights=return_PH) # alternate version already weighted
        actor_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits, labels=action_PH)
        actor_ce_mean = tf.reduce_mean(actor_ce)
        loss = tf.reduce_mean(return_PH * actor_ce) # return * policy(a|s)

    return {
        'observation_PH':   observation_PH,
        'return_PH':        return_PH,
        'action_PH':        action_PH,
        'probs':            probs,
        'actor_ce_mean':    actor_ce_mean,
        'loss':             loss,
        'zeroes':           zsL}