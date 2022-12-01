from pypaq.neuralmess.base_elements import tf
from pypaq.neuralmess.layers import lay_dense

def critic_graph(
        name=               'ac_critic',
        observation_width=  4,
        gamma=              0.99, # discount factor (gamma)
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
        action_OH_PH = tf.placeholder( # one-hot vector of action taken
            shape=  (None, num_actions),
            dtype=  tf.float32,
            name=   'action_OH_probs')
        next_action_qvs_PH = tf.placeholder(
            shape=  (None, num_actions),
            dtype=  tf.float32,
            name=   'next_action_qvs')
        next_action_probs_PH = tf.placeholder(
            shape=  (None, num_actions),
            dtype=  tf.float32,
            name=   'next_action_probs')
        reward_PH = tf.placeholder(
            shape=  None,
            dtype=  tf.float32,
            name=   'reward')

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

        # QVs (for all actions of current observation)
        qvs = lay_dense(
            input=      layer,
            name=       'qvs',
            units=      num_actions,
            activation= None,
            seed=       seed)

        qv =     tf.reduce_sum(qvs *                action_OH_PH,         axis=1) # Q(s,a)
        next_V = tf.reduce_sum(next_action_qvs_PH * next_action_probs_PH, axis=1) # V(next_s)
        labels = reward_PH + gamma * next_V
        loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=qv, labels=labels))
        if verb>0:
            print(f' > qvs: {qvs}')
            print(f' > qv: {qv}')
            print(f' > labels: {labels}')

    return {
        'observation_PH':       observation_PH,
        'action_OH_PH':         action_OH_PH,
        'next_action_qvs_PH':   next_action_qvs_PH,
        'next_action_probs_PH': next_action_probs_PH,
        'reward_PH':            reward_PH,
        'qvs':                  qvs,
        'loss':                 loss}