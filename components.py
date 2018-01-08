import tensorflow as tf


def get_weight_variable(name, shape, initializer=None, regularizer=None):
    if initializer is None:
        if len(shape) == 2:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer)
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, variable)
    return variable


def get_bias_variable(name, shape, initializer=None, regularizer=None):
    initializer = initializer or tf.constant_initializer(0.0)
    variable = tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer)
    return variable


def linear_layer(x, n_input, n_output, bias=True, variable_scope="linear"):
    """
    perform a linear transformation on x
    Args:
        x: [batch_size, n_input]
        n_input: input size
        n_output: output size
        bias: use bias or not
    Returns:
        y: a tensor with shape [batch_size, n_output]
    """
    with tf.variable_scope(variable_scope):
        W = get_weight_variable("W", [n_input, n_output])

        y = tf.matmul(x, W)
        if bias:
            b = get_bias_variable("b", [n_output])
            y = y + b
    return y

def batch_norm(x, decay=0.999, is_training=True, variable_scope="bn"):
    """Batch normalization
    Args:
        x: input tensor, shape [B, ...], x rank in [2, 3, 4]
        is_training: is training or not
    Returns:
        tensor with the same shape as x
    """
    with tf.variable_scope(variable_scope):
        x_rank = x.get_shape().ndims
        assert x_rank>=2 and x_rank<=4
        if x_rank == 2:
            x = tf.expand_dims(x, axis=1)
            x = tf.expand_dims(x, axis=2)
        elif x_rank == 3:
            x = tf.expand_dims(x, axis=1)

        y = tf.contrib.layers.batch_norm(
            inputs=x,
            decay=decay,
            center=True,
            scale=True,
            activation_fn=None,
            updates_collections=None,
            is_training=is_training,
            zero_debias_moving_mean=True,
            fused=True)
        if x_rank == 2:
            y = tf.squeeze(y, axis=[1, 2])
        elif x_rank == 3:
            y = tf.squeeze(y, axis=1)
    return y

