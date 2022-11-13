"""

 2022 (c) piteren

    NN TF graph elements and tools

"""

import tensorflow as tf
from typing import List



# default initializer for variables of graph (recommended by TF, BERT: stddev = 0.02)
def my_initializer(seed=12321, stddev=0.02):
    # TODO: consider GlorotUniform (Keras default) - compare on some cases
    return tf.keras.initializers.TruncatedNormal(stddev=stddev, seed=seed)

# GeLU (Gaussian Error Linear Unit) activation https://arxiv.org/abs/1606.08415
def gelu(x):
    # ga = x * 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))) # old version from TF.1.X
    return tf.keras.activations.gelu(x)

# replaces nan values @tensor with zero
def replace_nan_with_zero(tensor): return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)

# scales learning rate with warmUp and annealing (after warmUp)
def lr_scaler(
        baseLR,                            # initial learning rate
        g_step: tf.Tensor,              # global step
        warm_up: int or None=   1000,   # warmup steps, None or 0 turns-off
        ann_base: float=        0.999,  # annealing base, None or 1 for turn-off
        ann_step: float=        1.0,    # annealing step, higher value speeds up annealing
        n_wup_off: float=       2.0,    # N warmUp offset of annealing
        verb=                   0):

    if verb>0: print(f'\nbuilding lR scaling graph for baseLR: {baseLR} ..')
    g_step_fl = tf.cast(g_step, dtype=tf.float32)
    if warm_up is None: warm_up = 0
    lR = baseLR

    if warm_up:
        ratioWm = tf.reduce_min([g_step_fl, warm_up]) / warm_up # warmUp ratio
        lR = baseLR * ratioWm # learning rate with warmup
        if verb>0: print(f'applied warmUp ({warm_up}) to lR')
    if ann_base is not None and ann_base != 1:
        gStep_offs = tf.reduce_max([0, g_step_fl - warm_up * n_wup_off]) # offset by warmUpSteps
        lR *= ann_base ** (gStep_offs * ann_step) # learning rate with annealing
        if verb>0: print(f'applied annealing to lR ({ann_base:.5f},{ann_step:.5f})')
    return lR

# gradient clipping layer (by clip_value or AVT algorithm)
def grad_clipper_AVT(
        variables: List[tf.Variable],   # list of variables
        gradients: List[tf.Tensor],     # list of gradients (for variables)
        ggnorm_avt: tf.Variable,        # variable that holds averaged_over_time global_gradients_norm
        optimizer: tf.keras.optimizers.Optimizer,
        clip_value=         None,       # clipping value, for None clips with AVT
        avt_window: int=    100,        # width of averaging window (number of steps)
        avt_max_upd: float= 1.5,        # single step max factor of avt update
        do_clip=            True,       # disables clipping (just GN calculations)
        verb=               0):

    if verb > 0: print(f'grad_clipper_AVT: start value {ggnorm_avt}, avt_window {avt_window}, avt_max_upd {avt_max_upd:.1f}')

    ggnorm = tf.linalg.global_norm(gradients) # global norm of gradients

    avt_update = tf.reduce_min([ggnorm, avt_max_upd * ggnorm_avt]) # single value to update AVTG with (current GNorm or clipped to max value)

    # new value
    new_val = (ggnorm_avt * (avt_window - 1) + avt_update) / avt_window
    ggnorm_avt.assign(new_val)

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