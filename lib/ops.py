from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


import tensorflow.contrib.slim as slim
import pdb

# Define the convolution building block
def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=False, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                               activation_fn=None)


def conv3d(batch_input, depth,height,width, output_channel, stride, use_bias=True, scope='cov3d'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if use_bias:
            return slim.conv3d(batch_input, output_channel, [depth, height, width], stride, 'SAME',
                               data_format="NDHWC",
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv3d(batch_input, output_channel, [depth, height, width], stride, 'SAME',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                               data_format="NDHWC",
                               biases_initializer=None)


# Define our tensorflow version PRelu
def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5

    return pos + neg


# Define our Lrelu
def lrelu(inputs, alpha):
    return tf.nn.leaky_relu(inputs, alpha=alpha)


#Define the bathnorm layers
def batchnorm(inputs, is_training):
    return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                           scale=False, fused=True, is_training=is_training)

# Our dense layer
def denselayer(inputs, output_size):
    output = tf.layers.dense(inputs, output_size, activation=None,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    return output


# The implementation of PixelShuffler
def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output


def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)


# The random flip operation used for loading examples
def random_flip(input, decision):
    f1 = tf.identity(input)
    f2 = tf.image.flip_left_right(input)
    output = tf.cond(tf.less(decision, 0.5), lambda: f2, lambda: f1)

    return output


# The operation used to print out the configuration
def print_configuration_op(FLAGS):
    print('[Configurations]:')
    a = FLAGS.mode
    # pdb.set_trace()
    for name, value in FLAGS.__flags.items():
        if type(value) == float:
            print('\t%s: %f' % (name, value))
        elif type(value) == int:
            print('\t%s: %d' % (name, value))
        elif type(value) == str:
            print('\t%s: %s' % (name, value))
        elif type(value) == bool:
            print('\t%s: %s' % (name, value))
        else:
            print('\t%s: %s' % (name, value))

    print('End of configuration')


def compute_psnr(ref, target):
    ref = tf.cast(ref, tf.float32)
    target = tf.cast(target, tf.float32)
    diff = target - ref
    sqr = tf.multiply(diff, diff)
    err = tf.reduce_sum(sqr)
    v = tf.shape(diff)[0] * tf.shape(diff)[1] * tf.shape(diff)[2] * tf.shape(diff)[3]
    mse = err / tf.cast(v, tf.float32)
    psnr = 10. * (tf.log(255. * 255. / mse) / tf.log(10.))

    return psnr