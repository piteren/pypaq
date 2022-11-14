"""

 2020 (c) piteren

    baseline DQN graph for NEModel (TF)

"""

from pypaq.neuralmess.base_elements import tf
from pypaq.neuralmess.layers import lay_dense


def dqn_graph(
        name=               'qnn',
        num_actions: int=   4,
        observation_width=  4,
        hidden_layers=      (12,),
        seed=               121):

    with tf.variable_scope(name):

        observations_PH = tf.placeholder(   # observations input vector
            shape=  [None,observation_width],
            dtype=  tf.float32,
            name=   'observations')
        enum_actions_PH = tf.placeholder(   # enumerated action indexes (0,1),(1,3),(2,0),..
            shape=  [None,2],
            dtype=  tf.int32,
            name=   'enum_actions')
        gold_QV_PH = tf.placeholder(        # gold_QV
            shape=  [None],
            dtype=  tf.float32,
            name=   'gold_QV')

        layer = tf.keras.layers.LayerNormalization(axis=-1)(observations_PH)
        for i in range(len(hidden_layers)):
            layer = lay_dense(
                input=      layer,
                name=       f'hidden_layer_{i}',
                units=      hidden_layers[i],
                activation= tf.nn.relu,
                seed=       seed)
            layer = tf.keras.layers.LayerNormalization(axis=-1)(layer)

        # QV for all actions (for given input(state))
        output = lay_dense(
            input=      layer,
            units=      num_actions,
            activation= None,
            seed=       seed)

        pred_qv = tf.gather_nd(output, indices=enum_actions_PH)

        # loss on predicted vs next, we want predicted to match next
        loss = tf.losses.mean_squared_error(
            labels=         gold_QV_PH,
            predictions=    pred_qv,
            reduction=      tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE)

    return {
        'gold_QV_PH':       gold_QV_PH,
        'observations_PH':  observations_PH,
        'enum_actions_PH':  enum_actions_PH,
        'output':           output,
        'loss':             loss}