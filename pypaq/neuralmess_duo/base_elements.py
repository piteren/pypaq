"""

 2022 (c) piteren

    NN TF graph elements and tools

"""

import tensorflow as tf
from typing import List, Union, Optional



# default initializer for variables of graph (recommended by TF, BERT: stddev = 0.02)
def my_initializer(seed=12321, stddev=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=stddev, seed=seed)

# GeLU (Gaussian Error Linear Unit) activation https://arxiv.org/abs/1606.08415
def gelu(x):
    # ga = x * 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))) # old version from TF.1.X
    return tf.keras.activations.gelu(x)

# replaces nan values @tensor with zero
def replace_nan_with_zero(tensor): return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)

# gradient clipping layer (by clip_value or AVT algorithm)
def grad_clipper_AVT(
        variables: List[tf.Variable],   # list of variables
        gradients: List[tf.Tensor],     # list of gradients (for variables)
        ggnorm_avt: tf.Variable,        # variable that holds averaged_over_time global_gradients_norm
        optimizer: tf.keras.optimizers.Optimizer,
        clip_value=         None,       # clipping value, for None clips with AVT
        avt_SVal: float=    0.1,        # start value for AVT (smaller value makes warmup)
        avt_window: int=    100,        # width of averaging window (number of steps)
        avt_max_upd: float= 1.5,        # single step max factor of avt update
        do_clip=            True,       # disables clipping (just GN calculations)
        verb=               0):

    ggnorm = tf.linalg.global_norm(gradients) # global norm of gradients

    avt_update = tf.reduce_min([ggnorm, avt_max_upd * ggnorm_avt]) # single value to update AVTG with (current GNorm or clipped to max value)

    # new value
    new_val = (ggnorm_avt * (avt_window - 1) + avt_update) / avt_window
    ggnorm_avt.assign(new_val)
    if verb>0: print(f'grad_clipper_AVT: avt_SVal {avt_SVal:.1f}, avt_window {avt_window}, avt_max_upd {avt_max_upd:.1f}')

    if do_clip:
        gradients, _ = tf.clip_by_global_norm(
            t_list=     gradients,
            clip_norm=  clip_value or ggnorm_avt,
            use_norm=   ggnorm)
        if verb>0: print(f' >> is clipping gradients {"with value" if clip_value else "with AVT"}')
    elif verb>0: print(' >> not doing clipping')

    optimizer.apply_gradients(grads_and_vars=zip(gradients,variables))

    return {
        'gradients':    gradients,
        'ggnorm':       ggnorm}