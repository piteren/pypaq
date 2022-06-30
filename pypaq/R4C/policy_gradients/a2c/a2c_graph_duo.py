import tensorflow as tf

from pypaq.neuralmess_duo.layers import lay_dense


def a2c_graph_duo(
        observation_width=  4,
        num_actions=        2,
        hidden_layers=      (128,),
        lay_norm=           False,
        seed=               123,
        verb=               0):

    observation = tf.keras.Input(shape=(observation_width,), name="observation")
    action =      tf.keras.Input(shape=(1,),                 name="action")
    ret =         tf.keras.Input(shape=(1,),                 name="ret")

    layer = observation
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
    if verb>0: print(f' > value: {value}')

    advantage = ret - value
    if verb>0: print(f' > advantage: {advantage}')

    # action logits of Actor policy
    action_logits = lay_dense(
        input=      layer,
        name=       'action_logits',
        units=      num_actions,
        activation= None,
        seed=       seed)
    if verb>0: print(f' > action_logits: {action_logits}')

    action_prob = tf.nn.softmax(action_logits)
    max_probs = tf.reduce_max(action_prob, axis=-1) # max action_probs
    min_probs = tf.reduce_min(action_prob, axis=-1) # min action_probs
    amax_prob = tf.reduce_mean(max_probs) # average of batch max action_prob
    amin_prob = tf.reduce_mean(min_probs) # average of batch min action_prob

    # ************************************************************************************************** loss definition

    actor_ce = tf.keras.metrics.sparse_categorical_crossentropy(
        y_true=         action,
        y_pred=         action_logits,
        from_logits=    True)
    if verb>0: print(f' > actor_ce: {actor_ce}')
    actor_ce_mean = tf.reduce_mean(actor_ce)
    loss_actor_weighted_mean = tf.reduce_mean(actor_ce * advantage)

    loss_critic = tf.keras.losses.huber(
        y_true=         ret,
        y_pred=         value)
    if verb>0: print(f' > loss_critic: {loss_critic}')
    loss_critic_mean = tf.reduce_mean(loss_critic)

    loss = loss_actor_weighted_mean + loss_critic_mean
    if verb>0: print(f' > loss: {loss}')

    return {
        'observation':      observation,
        'action':           action,
        'ret':              ret,

        'value':            value,
        'advantage':        advantage,
        'action_logits':    action_logits,
        'action_prob':      action_prob,
        'amax_prob':        amax_prob,
        'amin_prob':        amin_prob,
        'actor_ce_mean':    actor_ce_mean,

        'loss':             loss,
        'loss_actor':       loss_actor_weighted_mean,
        'loss_critic':      loss_critic_mean,

        'train_model_IO':   {
            'inputs':       ['observation','action','ret'],
            'outputs':      ['actor_ce_mean','amax_prob','amin_prob','loss_actor','loss_critic']},
    }


if __name__ == '__main__':

    a2c_graph_duo(verb=1)