import tensorflow as tf
import numpy as np
import tensorflow as tf
from contextlib import contextmanager

def gated_linear_layer(inputs, gates, name=None):
    activation = tf.multiply(x=inputs, y=tf.sigmoid(gates), name=name)

    return activation


def instance_norm_layer(
        inputs,
        epsilon=1e-06,
        activation_fn=None,
        name=None):
    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs=inputs,
        epsilon=epsilon,
        activation_fn=activation_fn)

    return instance_norm_layer


def conv1d_layer(
        inputs,
        filters,
        kernel_size,
        strides=1,
        padding='same',
        activation=None,
        kernel_initializer=None,
        name=None):
    conv_layer = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name)

    return conv_layer


def conv2d_layer(
        inputs,
        filters,
        kernel_size,
        strides,
        padding='same',
        activation=None,
        kernel_initializer=None,
        name=None):
    conv_layer = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name)

    return conv_layer


def residual1d_block(
        inputs,
        filters=512,
        kernel_size=3,
        strides=1,
        name_prefix='residule_block_'):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')
    h2 = conv1d_layer(inputs=h1_glu, filters=filters // 2, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs=h2, activation_fn=None, name=name_prefix + 'h2_norm')

    h3 = inputs + h2_norm

    return h3


def downsample1d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        name_prefix='downsample1d_block_'):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def downsample2d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        name_prefix='downsample2d_block_'):
    h1 = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def upsample1d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        shuffle_size=2,
        name_prefix='upsample1d_block_'):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs=h1, shuffle_size=shuffle_size, name=name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs=h1_shuffle, activation_fn=None, name=name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs=h1_gates, shuffle_size=shuffle_size, name=name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_shuffle_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def upsample2d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        shuffle_size=2,
        name_prefix='upsample1d_block_'):
    # h1 = tf.layers.conv2d_transpose(inputs, 256, [5, 5], strides=[2, 2], activation=None, padding="same")
    h1 = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                      name=name_prefix + 'h1_conv')
    # h1_shuffle = pixel_2Dshuffler(inputs=h1, shuffle_size=shuffle_size, name=name_prefix + 'h1_shuffle')
    h1_shuffle = up_2Dsample(x=h1, scale_factor=shuffle_size)
    h1_norm = instance_norm_layer(inputs=h1_shuffle, activation_fn=None, name=name_prefix + 'h1_norm')

    h1_gates = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                            name=name_prefix + 'h1_gates')
    # h1_shuffle_gates = pixel_2Dshuffler(inputs=h1_gates, shuffle_size=shuffle_size, name=name_prefix + 'h1_shuffle_gates')
    h1_shuffle_gates = up_2Dsample(x=h1_gates, scale_factor=shuffle_size)
    h1_norm_gates = instance_norm_layer(inputs=h1_shuffle_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu



def up_2Dsample(x, scale_factor=2):
    # _, h, w, _ = x.get_shape().as_list()
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)


def pixel_shuffler(inputs, shuffle_size=2, name=None):
    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor=inputs, shape=[n, ow, oc], name=name)

    return outputs


def pixel_2Dshuffler(inputs, shuffle_size=2, name=None):
    n = tf.shape(inputs)[0]
    w1 = tf.shape(inputs)[1]
    w2 = tf.shape(inputs)[2]
    c = inputs.get_shape().as_list()[3]

    oc = c // shuffle_size // shuffle_size
    ow1 = w1 * shuffle_size
    ow2 = w2 * shuffle_size

    outputs = tf.reshape(tensor=inputs, shape=[n, ow1, ow2, oc], name=name)

    return outputs


def generator_gated2Dcnn(inputs, reuse = False, scope_name = 'generator_gated2Dcnn'):
    res_filter = 512   
    inputs = tf.transpose(inputs, perm=[0, 2, 1], name='input_transpose')
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [5, 15], strides = [1, 1], activation = None, name = 'h1_conv')
        h1_gates = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [5, 15], strides = [1, 1], activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample2d_block(inputs = h1_glu, filters = 256, kernel_size = [5, 5], strides = [2, 2], name_prefix = 'downsample1d_block1_')
        d2 = downsample2d_block(inputs = d1, filters = 512, kernel_size = [5, 5], strides = [2, 2], name_prefix = 'downsample1d_block2_')

        # reshape : cyclegan-VC2
        d3 = tf.reshape(d2, shape=[tf.shape(d2)[0], -1, d2.get_shape()[3].value])
        # modification in paper - 2019.05.01
        d3 = conv1d_layer(inputs=d3, filters=256, kernel_size = 1, strides = 1, activation = None, name = '1x1_down_conv1d')
        d3 = instance_norm_layer(inputs=d3, activation_fn=None, name='d3_norm')

        # Residual blocks
        r1 = residual1d_block(inputs = d3, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_')
        r2 = residual1d_block(inputs = r1, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_')
        r3 = residual1d_block(inputs = r2, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_')
        r4 = residual1d_block(inputs = r3, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_')
        r5 = residual1d_block(inputs = r4, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_')
        r6 = residual1d_block(inputs = r5, filters = res_filter, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_')

        # filters从论文中的2304改成了512，为了避免超显存。
        r6 = conv1d_layer(r6, filters = 512, kernel_size = 1, strides = 1, activation = None, name = '1x1_up_conv1d')
        r6 = instance_norm_layer(inputs=r6, activation_fn=None, name='r6_norm')    
        r6 = tf.reshape(r6, shape=[tf.shape(d2)[0], tf.shape(d2)[1], tf.shape(d2)[2], d2.get_shape()[3].value])

        # Upsample
        u1 = upsample2d_block(inputs = r6, filters = 1024, kernel_size = [5, 5], strides = [1, 1], shuffle_size = 2, name_prefix = 'upsample1d_block1_')
        u2 = upsample2d_block(inputs = u1, filters = 512, kernel_size = [5, 5], strides = [1, 1], shuffle_size = 2, name_prefix = 'upsample1d_block2_')

        #filters从35降到1，为了避免超过显存。
        o1 = conv2d_layer(inputs = u2, filters = 1, kernel_size = [5, 15], strides = [1, 1], activation = None, name = 'o1_conv')
        # o1 = tf.squeeze(o1)
        o1 = tf.reshape(o1, shape=[tf.shape(o1)[0], tf.shape(o1)[1], -1])
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')

    return o2



def discriminator_2D(inputs, reuse = False, scope_name = 'discriminator'):
    #### 结构与原论文相同
    # inputs has shape [batch_size, num_features, time]
    # we need to add channel for 2D convolution [batch_size, num_features, time, 1]
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], strides = [1, 1], activation = None, name = 'h1_conv')
        h1_gates = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], strides = [1, 1], activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample2d_block(inputs = h1_glu, filters = 256, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block1_')
        d2 = downsample2d_block(inputs = d1, filters = 512, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block2_')
        d3 = downsample2d_block(inputs = d2, filters = 1024, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block3_')
        d4 = downsample2d_block(inputs=d3, filters=1024, kernel_size=[1, 5], strides=[1, 1], name_prefix='downsample2d_block4_')


        # Output
        o1 = conv2d_layer(inputs=d4, filters=1, kernel_size=[1, 3], strides=[1, 1], activation=tf.nn.sigmoid, name='out_1d_conv')
        return o1
