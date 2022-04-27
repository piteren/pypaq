"""

 2022 (c) piteren

    NN TF graph elements and tools

"""

import tensorflow as tf



# default initializer for variables of graph (recommended by TF, BERT: stddev = 0.02)
def my_initializer(seed=12321, stddev=0.02):
    return tf.keras.initializers.TruncatedNormal(stddev=stddev, seed=seed)

# GeLU (Gaussian Error Linear Unit) activation https://arxiv.org/abs/1606.08415
def gelu(x):
    # ga = x * 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))) # old version from TF.1.X
    return tf.keras.activations.gelu(x)

# replaces nan values @tensor with zero
def replace_nan_with_zero(tensor): return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)