import tensorflow as tf

from pypaq.neuralmess_duo.base_elements import my_initializer


# dense layer wrap, wraps with call, adds default initializer
def lay_dense(
        input,
        units :int,                 # layer width
        name=           'dense',
        activation=     None,
        use_bias=       True,
        initializer=    None,
        seed=           12321):
    if initializer is None: initializer = my_initializer(seed)
    dense_lay = tf.keras.layers.Dense(
        units=              units,
        activation=         activation,
        use_bias=           use_bias,
        kernel_initializer= initializer,
        name=               name)
    output = dense_lay(input)
    return output