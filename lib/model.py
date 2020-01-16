# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.ops import *
import collections
import scipy.io as scio
import numpy as np

#load training data
def load_data(Flags):
    #Need to Modify the path of training data
    data_HR = scio.loadmat(Flags.input_dir_HR)
    data_LR = scio.loadmat(Flags.input_dir_LR)
    HR = data_HR['train']
    LR = data_LR['train_ds']
    LR = np.expand_dims(LR, axis=4)
    LR = np.transpose(LR, axes=[0, 3, 1, 2, 4])

    HR = HR.astype(np.float32)
    LR = LR.astype(np.float32)

    output = tf.train.slice_input_producer([LR, HR], num_epochs=None, shuffle=True)

    inputs_batch, targets_batch = tf.train.shuffle_batch(
        [output[0], output[1]], batch_size=1, capacity=5,
        min_after_dequeue=2, num_threads=4)

    return inputs_batch, targets_batch


# "reuse" means wether use the same scope name in the net,default setting is False
# Definition of the generator without any bach normalization layers
def generator(gen_inputs, gen_output_channels, reuse=False, FLAGS=None):
    # Check the flag
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    # The Bx residual blocks
    # Modify the blocks
    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope, reuse=reuse):
            net = conv3d(inputs, 5, 3, 3, output_channel, stride, use_bias=False, scope='conv3d_1')
            net = prelu_tf(net)
            net = conv3d(net, 5, 3, 3, output_channel, stride, use_bias=False, scope='conv3d_2')
            net = net + inputs

        return net

    with tf.variable_scope('generator_unit', reuse=reuse):
        residual_num = 0
        # The input layer
        with tf.device("/gpu:0"):
            with tf.variable_scope('input_stage'):
                net = conv3d(gen_inputs, 5, 3, 3, 32, 1, use_bias=False, scope='conv')
                net = prelu_tf(net)

                stage1_output = net

            # The half of the residual block parts
            for i in range(1, FLAGS.num_resblock + 1, 1):
                name_scope = 'resblock_%d' % (i)
                net = residual_block(net, 32, 1, name_scope)

            with tf.variable_scope('resblock_output'):
                net = conv3d(net, 5, 3, 3, 32, 1, use_bias=False, scope='conv')

            # skip connect
            net = net + stage1_output

            with tf.variable_scope('resblock_output_squeeze_and_transpose'):
                net = conv3d(net, 5, 3, 3, 1, 1, use_bias=False, scope='dense_channel')
                net = tf.squeeze(net, axis=4)
                net = tf.transpose(net, perm=[0, 2, 3, 1])

            with tf.variable_scope('subpixelconv_stage'):
                net = conv2(net, 3, 256, 1, scope='conv')
                net = pixelShuffler(net, scale=2)
                net = prelu_tf(net)

            with tf.variable_scope('output_stage'):
                net = conv2(net, 9, gen_output_channels, 1, scope='conv')

    return net

# Definition of the discriminator
def discriminator(dis_inputs, FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            net = conv2(inputs, kernel_size, output_channel, stride, use_bias=True, scope='conv1')
            net = batchnorm(net, FLAGS.is_training)
            net = lrelu(net, 0.2)
        return net

    with tf.device('/gpu:0'):
        with tf.variable_scope('discriminator_unit'):
            # The input layer
            with tf.variable_scope('input_stage'):
                net = conv2(dis_inputs, 3, 32, 1, scope='conv')
                net = lrelu(net, 0.2)

                # The discriminator block part
                # block 1
                net = discriminator_block(net, 32, 3, 2, 'disblock_1')

                # block 2
                net = discriminator_block(net, 64, 3, 1, 'disblock_2')

                # block 3
                net = discriminator_block(net, 64, 3, 2, 'disblock_3')

                # block 4
                net = discriminator_block(net, 128, 3, 1, 'disblock_4')

                # block 5
                net = discriminator_block(net, 128, 3, 2, 'disblock_5')

                # block 6
                net = discriminator_block(net, 256, 3, 1, 'disblock_6')

                # block 7
                net = discriminator_block(net, 256, 3, 2, 'disblock_7')

            # The dense layer 1
            with tf.variable_scope('dense_layer_1', reuse=tf.AUTO_REUSE):
                net = slim.flatten(net)
                net = denselayer(net, 1024)
                net = lrelu(net, 0.2)

            # The dense layer 2
            with tf.variable_scope('dense_layer_2', reuse=tf.AUTO_REUSE):
                net = denselayer(net, 1)
                net = tf.nn.sigmoid(net)

    return net

    # Define the whole network architecture


def _3DSRGAN(inputs, targets, FLAGS):
    # Define the container of the parameter
    Network = collections.namedtuple('Network', 'discrim_real_output, discrim_fake_output, discrim_loss, spectral_loss \
    discrim_grads_and_vars, adversarial_loss, content_loss, gen_grads_and_vars, gen_output, train, global_step, \
    learning_rate,gen_loss')

    # Build the generator part
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output = generator(inputs, output_channel, reuse=False, FLAGS=FLAGS)

    # Build the fake discriminator
    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse=False):
            discrim_fake_output = discriminator(gen_output, FLAGS=FLAGS)

    # Build the real discriminator
    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator', reuse=False):
            discrim_real_output = discriminator(targets, FLAGS=FLAGS)

    # Use MSE loss directly
    # extract the feature map of Generator Output and Real Output
    if FLAGS.perceptual_mode == 'MSE':
        extracted_feature_gen = gen_output
        extracted_feature_target = targets
    else:
        raise NotImplementedError('Unknown perceptual type!!')

    # Calculating the generator loss
    with tf.variable_scope('generator_loss', reuse=tf.AUTO_REUSE):
        # Content loss
        with tf.variable_scope('content_loss', reuse=tf.AUTO_REUSE):
            # Compute the euclidean distance between the two features
            diff = extracted_feature_gen - extracted_feature_target
            # if FLAGS.perceptual_mode == 'MSE':
            content_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(diff), axis=[3])))

        with tf.variable_scope('spectral_loss', reuse=tf.AUTO_REUSE):
            gen_dis = tf.sqrt(tf.reduce_sum(extracted_feature_gen * extracted_feature_gen, axis=3))
            target_dis = tf.sqrt(tf.reduce_sum(extracted_feature_target * extracted_feature_target, axis=3))
            temp = tf.reduce_sum(extracted_feature_target * extracted_feature_gen, axis=3)
            spectral_loss = tf.reduce_mean(tf.math.acos(temp / (gen_dis * target_dis)))

            with tf.variable_scope('adversarial_loss', reuse=tf.AUTO_REUSE):
                # dont use 1-log(D(G(I'lr)))
                adversarial_loss = tf.reduce_mean(-tf.log(discrim_fake_output + FLAGS.EPS))

            gen_loss = content_loss + spectral_loss + adversarial_loss

            # Calculating the discriminator loss
            with tf.variable_scope('discriminator_loss', reuse=tf.AUTO_REUSE):
                discrim_fake_loss = tf.log(1 - discrim_fake_output + FLAGS.EPS)
                discrim_real_loss = tf.log(discrim_real_output + FLAGS.EPS)
                discrim_loss = tf.reduce_mean(-(discrim_fake_loss + discrim_real_loss))

            # Define the learning rate and global step
            with tf.variable_scope('get_learning_rate_and_global_step', reuse=tf.AUTO_REUSE):
                global_step = tf.contrib.framework.get_or_create_global_step()
                learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step,
                                                           FLAGS.decay_rate,
                                                           staircase=FLAGS.stair)
                incr_global_step = tf.assign(global_step, global_step + 1)

            with tf.variable_scope('dicriminator_train', reuse=tf.AUTO_REUSE):
                discrim_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
                discrim_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
                discrim_grads_and_vars = discrim_optimizer.compute_gradients(discrim_loss, discrim_tvars)
                discrim_train = discrim_optimizer.apply_gradients(discrim_grads_and_vars)

            with tf.variable_scope('generator_train', reuse=tf.AUTO_REUSE):
                # Need to wait discriminator to perform train step
                with tf.control_dependencies([discrim_train] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
                    gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
                    gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
                    gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

                    #use moving average on loss??
                    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
                    update_loss = exp_averager.apply([discrim_loss, adversarial_loss, content_loss, spectral_loss])

        return Network(
            discrim_real_output=discrim_real_output,
            discrim_fake_output=discrim_fake_output,
            discrim_loss=exp_averager.average(discrim_loss),
            discrim_grads_and_vars=discrim_grads_and_vars,
            adversarial_loss=exp_averager.average(adversarial_loss),
            content_loss=exp_averager.average(content_loss),
            spectral_loss=exp_averager.average(spectral_loss),
            gen_grads_and_vars=gen_grads_and_vars,
            gen_output=gen_output,
            train=tf.group(update_loss, incr_global_step, gen_train),
            global_step=global_step,
            learning_rate=learning_rate,
            gen_loss=gen_loss
        )


def _3DSRResnet(inputs, targets, FLAGS):
    # Define the container of the parameter
    Network = collections.namedtuple('Network', 'content_loss, gen_grads_and_vars, gen_output, train, global_step, \
        learning_rate,gen_loss')

    # Build the generator part
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output = generator(inputs, output_channel, reuse=False, FLAGS=FLAGS)

    if FLAGS.perceptual_mode == 'MSE':
        extracted_feature_gen = gen_output
        extracted_feature_target = targets

    else:
        raise NotImplementedError('Unknown perceptual type')

    # Calculating the generator loss
    with tf.variable_scope('generator_loss', reuse=tf.AUTO_REUSE):
        # Content loss
        with tf.variable_scope('content_loss', reuse=tf.AUTO_REUSE):
            # Compute the euclidean distance between the two features
            # check=tf.equal(extracted_feature_gen, extracted_feature_target)
            diff = extracted_feature_gen - extracted_feature_target
            # if FLAGS.perceptual_mode == 'MSE':
            content_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(diff), axis=3)))

        gen_loss = content_loss

    # Define the learning rate and global step
    with tf.variable_scope('get_learning_rate_and_global_step', reuse=tf.AUTO_REUSE):
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step,
                                                   FLAGS.decay_rate,
                                                   staircase=FLAGS.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('generator_train', reuse=tf.AUTO_REUSE):
        # Need to wait discriminator to perform train step
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
            gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
            gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

    #use moving average on loss??
    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_averager.apply([content_loss])

    return Network(
        content_loss=exp_averager.average(content_loss),
        gen_grads_and_vars=gen_grads_and_vars,
        gen_output=gen_output,
        train=tf.group(update_loss, incr_global_step, gen_train),
        global_step=global_step,
        learning_rate=learning_rate,
        gen_loss=gen_loss
    )
