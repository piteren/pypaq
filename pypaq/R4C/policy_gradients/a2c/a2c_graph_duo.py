import tensorflow as tf

from pypaq.neuralmess_duo.layers import lay_dense


def a2c_graph_duo(
        observation_width=  4,
        num_actions=        2,
        two_towers=         False,  # builds separate towers for Actor & Critic
        hidden_layers=      (128,),
        lay_norm=           False,
        use_scaled_ce=      True,   # for True uses experimental scaled ce loss for Actor
        use_huber=          False,  # for True uses Huber loss for Critic
        seed=               123,
        verb=               0):

    observation = tf.keras.Input(shape=(observation_width,), name="observation")
    action =      tf.keras.Input(shape=(1,),                 name="action", dtype=tf.int32)
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

    layer_second_tower = layer
    if two_towers:
        layer_second_tower = observation
        if lay_norm: layer = tf.keras.layers.LayerNormalization(axis=-1)(layer)
        for i in range(len(hidden_layers)):
            layer_second_tower = lay_dense(
                input=      layer_second_tower,
                name=       f'hidden_layer_st_{i}',
                units=      hidden_layers[i],
                activation= tf.nn.relu,
                seed=       seed)
            if lay_norm: layer_second_tower = tf.keras.layers.LayerNormalization(axis=-1)(layer_second_tower)

    # Value from critic
    value = lay_dense(
        input=      layer_second_tower,
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

    if use_scaled_ce:

        action_prob_selected = tf.gather(params=action_prob, indices=action, axis=1, batch_dims=1)
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

        actor_ce = tf.keras.metrics.sparse_categorical_crossentropy(
            y_true=         action,
            y_pred=         action_logits,
            from_logits=    True)

        loss_actor_weighted_mean = tf.reduce_mean(actor_ce * advantage)

    if verb>0: print(f' > actor_ce: {actor_ce}')
    actor_ce_mean = tf.reduce_mean(actor_ce)

    if use_huber:
        loss_critic = tf.keras.losses.huber(y_true=ret, y_pred=value)
    else:
        loss_critic = (ret-value)*(ret-value) # == MSE (advantage^2)

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
        'actor_ce':         actor_ce,
        'actor_ce_mean':    actor_ce_mean,

        'loss':             loss,
        'loss_actor':       loss_actor_weighted_mean,
        'loss_critic':      loss_critic_mean,

        'train_model_IO':   {
            'inputs':       ['observation','action','ret'],
            'outputs':      ['actor_ce_mean','amax_prob','amin_prob','loss_actor','loss_critic','advantage','value','action_prob','actor_ce']},
    }


if __name__ == '__main__':

    a2c_graph_duo(verb=1)