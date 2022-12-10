from typing import List

from pypaq.neuralmess.base_elements import tf
from pypaq.neuralmess.layers import lay_dense, zeroes


def a2c_graph(
        name=               'a2c',
        observation_width=  4,
        num_actions=        2,
        two_towers=         False,  # builds separate towers for Actor & Critic
        num_layers=         1,
        layer_width=        50,
        lay_norm=           False,
        use_scaled_ce=      True,   # for True uses experimental scaled ce loss for Actor
        use_huber=          False,  # for True uses Huber loss for Critic
        seed=               123,
        verb=               0):

    with tf.variable_scope(name):

        observation_PH = tf.placeholder(
            shape=  (None, observation_width),
            dtype=  tf.float32,
            name=   'observation')

        action_PH = tf.placeholder(
            shape=  None,
            dtype=  tf.int32,
            name=   'action')

        return_PH = tf.placeholder(
            shape=  None,
            dtype=  tf.float32,
            name=   'return')

        zsL: List[tf.Tensor] = []  # list of zeroes Tensors

        hidden_layers = tuple([layer_width] * num_layers)

        inp = tf.keras.layers.LayerNormalization(axis=-1)(observation_PH) if lay_norm else observation_PH

        layer = inp
        for i in range(len(hidden_layers)):
            layer = lay_dense(
                input=      layer,
                name=       f'hidden_layer_{i}',
                units=      hidden_layers[i],
                activation= tf.nn.relu,
                seed=       seed)
            zsL.append(zeroes(layer))
            if lay_norm: layer = tf.keras.layers.LayerNormalization(axis=-1)(layer)

        layer_second_tower = layer
        if two_towers:
            layer_second_tower = inp
            for i in range(len(hidden_layers)): # INFO: shape of second tower will be as first one
                layer_second_tower = lay_dense(
                    input=      layer_second_tower,
                    name=       f'hidden_layer_st_{i}',
                    units=      hidden_layers[i],
                    activation= tf.nn.relu,
                    seed=       seed)
                zsL.append(zeroes(layer_second_tower))
                if lay_norm: layer_second_tower = tf.keras.layers.LayerNormalization(axis=-1)(layer_second_tower)

        # Value from critic
        value = lay_dense(
            input=      layer_second_tower,
            name=       'value',
            units=      1,
            activation= None,
            seed=       seed)
        if verb > 0: print(f' > value: {value}')

        advantage = return_PH - value
        if verb > 0: print(f' > advantage: {advantage}')

        # action logits of Actor policy
        action_logits = lay_dense(
            input=      layer,
            name=       'action_logits',
            units=      num_actions,
            activation= None,
            seed=       seed)
        if verb > 0: print(f' > action_logits: {action_logits}')
        probs = tf.nn.softmax(action_logits)

        # ************************************************************************************************** loss definition

        if use_scaled_ce:

            action_prob_selected = tf.gather(params=probs, indices=action_PH, axis=1, batch_dims=1)
            #action_prob_selected = tf.squeeze(action_prob_selected)
            if verb>0: print(f' > action_prob_selected: {action_prob_selected}')

            # crossentropy loss
            actor_ce = -tf.math.log(action_prob_selected)
            actor_ce_neg = -tf.math.log(1-action_prob_selected)

            # select loss for positive and negative advantage
            actor_ce = tf.where(
                condition=  tf.greater(advantage,0),
                x=          actor_ce,
                y=          actor_ce_neg)

            loss_actor_weighted_mean = tf.reduce_mean(actor_ce * tf.math.abs(advantage))  # scale loss with advantage

        else:

            actor_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits, labels=action_PH)
            loss_actor_weighted_mean = tf.reduce_mean(actor_ce * advantage)

        if verb > 0: print(f' > actor_ce: {actor_ce}')
        actor_ce_mean = tf.reduce_mean(actor_ce)

        if use_huber:
            huber_loss = tf.keras.losses.Huber()
            loss_critic = huber_loss(value, return_PH)
        else:
            loss_critic = advantage * advantage # MSE

        if verb > 0: print(f' > loss_critic: {loss_critic}')
        loss_critic_mean = tf.reduce_mean(loss_critic)

        loss = loss_actor_weighted_mean + loss_critic_mean
        if verb > 0: print(f' > loss: {loss}')

    return {
        'observation_PH':   observation_PH,
        'action_PH':        action_PH,
        'return_PH':        return_PH,

        'value':            value,
        'advantage':        advantage,
        'action_logits':    action_logits,
        'probs':            probs,

        'actor_ce':         actor_ce,
        'actor_ce_mean':    actor_ce_mean,

        'loss':             loss,
        'loss_actor':       loss_actor_weighted_mean,
        'loss_critic':      loss_critic_mean,

        'zeroes':           zsL}