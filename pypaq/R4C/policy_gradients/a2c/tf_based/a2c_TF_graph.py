from pypaq.neuralmess.base_elements import tf
from pypaq.neuralmess.layers import lay_dense

def a2c_graph(
        name=               'a2c',
        observation_width=  4,
        num_actions=        2,
        hidden_layers=      (128,),
        lay_norm=           False,
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

        layer = observation_PH
        if lay_norm: layer = tf.keras.layers.LayerNormalization(axis=-1)(layer)
        for i in range(len(hidden_layers)):
            layer = lay_dense(
                input=      layer,
                name=       f'hidden_layer_{i}',
                units=      hidden_layers[i],
                activation= tf.nn.relu,
                seed=       seed)
            if lay_norm: layer = tf.keras.layers.LayerNormalization(axis=-1)(layer)

        # Value from critic
        value = lay_dense(
            input=      layer,
            name=       'value',
            units=      1,
            activation= None,
            seed=       seed)

        # action logits of Actor policy
        action_logits = lay_dense(
            input=      layer,
            name=       'action_logits',
            units=      num_actions,
            activation= None,
            seed=       seed)
        action_prob = tf.nn.softmax(action_logits)

        max_probs = tf.reduce_max(action_prob, axis=-1) # max action_probs
        min_probs = tf.reduce_min(action_prob, axis=-1) # min action_probs
        amax_prob = tf.reduce_mean(max_probs) # average of batch max action_prob
        amin_prob = tf.reduce_mean(min_probs) # average of batch min action_prob

        actor_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits, labels=action_PH)
        actor_ce_mean = tf.reduce_mean(actor_ce)
        advantage = return_PH - value
        loss_actor = tf.reduce_mean(advantage * actor_ce)

        huber_loss = tf.keras.losses.Huber()
        loss_critic = tf.reduce_mean(huber_loss(value, return_PH))

        loss = loss_actor + loss_critic

        if verb>0:
            print('AC_shared graph debug:')
            print(f' > value: {value}')
            print(f' > action_logits: {action_logits}')

    return {
        'observation_PH':   observation_PH,
        'action_PH':        action_PH,
        'return_PH':        return_PH,

        'action_prob':      action_prob,
        'value':            value,

        'amax_prob':        amax_prob,
        'amin_prob':        amin_prob,

        'actor_ce_mean':    actor_ce_mean,
        'loss_actor':       loss_actor,
        'loss_critic':      loss_critic,
        'loss':             loss}