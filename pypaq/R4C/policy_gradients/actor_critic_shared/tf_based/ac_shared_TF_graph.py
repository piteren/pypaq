from pypaq.neuralmess.base_elements import tf
from pypaq.neuralmess.layers import lay_dense

def acs_graph(
        name=               'acs',
        observation_width=  4,
        num_actions=        2,
        hidden_layers=      (24,24),
        lay_norm=           False,
        seed=               123,
        verb=               0):

    with tf.variable_scope(name):

        observation_PH = tf.placeholder(
            shape=  (None, observation_width),
            dtype=  tf.float32,
            name=   'observation')

        action_PH = tf.placeholder( # action taken (may differ from Actor policy action because of Trainer policy like exploration)
            shape=  None,
            dtype=  tf.int32,
            name=   'action')

        qv_label_PH = tf.placeholder( # label of Q(s,a), computed from: reward + gamma*V_next_action
            shape=  None,
            dtype=  tf.float32,
            name=   'next_action_qvs')

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

        # critic QVs (for all actions of current observation)
        qvs = lay_dense(
            input=      layer,
            name=       'qvs',
            units=      num_actions,
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

        value = tf.reduce_sum(qvs * action_prob)  # value of observation (next_observation)

        max_probs = tf.reduce_max(action_prob, axis=-1) # max action_probs
        min_probs = tf.reduce_min(action_prob, axis=-1) # min action_probs
        amax_prob = tf.reduce_mean(max_probs) # average of batch max action_prob
        amin_prob = tf.reduce_mean(min_probs) # average of batch min action_prob

        action_OH = tf.one_hot(action_PH, num_actions)
        qv = tf.reduce_sum(qvs*action_OH, axis=1) # Q(s,a)
        log_policy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits, labels=action_PH)
        loss_actor = tf.reduce_mean(qv * log_policy)

        #next_V = tf.reduce_sum(next_action_qvs_PH * next_action_probs_PH, axis=1) # V(next_s)
        #labels = reward_PH + gamma * next_V
        # TODO: try with hubner loss
        loss_critic = tf.reduce_mean(tf.losses.mean_squared_error(predictions=qv, labels=qv_label_PH))

        loss = loss_actor + loss_critic

        if verb>0:
            print('AC_shared graph debug:')
            print(f' > qvs: {qvs}')
            print(f' > action_logits: {action_logits}')
            print(f' > value: {value}')

    return {
        'observation_PH':   observation_PH,
        'action_PH':        action_PH,
        'qv_label_PH':      qv_label_PH,

        'qvs':              qvs,
        'action_prob':      action_prob,
        'value':            value,

        'amax_prob':        amax_prob,
        'amin_prob':        amin_prob,

        'loss_actor':       loss_actor,
        'loss_critic':      loss_critic,
        'loss':             loss}