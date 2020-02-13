from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from lib.model import generator, load_data, _3DSRGAN, _3DSRResnet
from lib.ops import *
import math
import time
import numpy as np
import scipy
import scipy.io as scio
from random import randint
import time

Flags = tf.app.flags

# The system parameter
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
Flags.DEFINE_string('checkpoint', './0909__3DSRResnet_01/model-43000',
                    'If provided, the weight will be restored from the provided checkpoint')
Flags.DEFINE_boolean('pre_trained_model', True,
                     'If set True, the weight will be loaded but the global_step will still '
                     'be 0. If set False, you are going to continue the training. That is, '
                     'the global_step will be initiallized from the checkpoint, too')
Flags.DEFINE_string('pre_trained_model_type', '_3DSRResnet', 'The type of pretrained model (_3DSRGAN or _3DSRResnet)')
Flags.DEFINE_boolean('is_training', True, 'Training => True, Testing => False')
Flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'path to checkpoint file for the vgg19')
Flags.DEFINE_string('task', None, 'The task: _3DSRGAN, _3DSRResnet')
# The data preparing operation
Flags.DEFINE_integer('batch_size', 16, 'Batch size of the input batch')
Flags.DEFINE_string('input_dir_LR', None, 'The directory of the input resolution input data')
Flags.DEFINE_string('input_dir_HR', None, 'The directory of the high resolution input data')
Flags.DEFINE_boolean('flip', True, 'Whether random flip data augmentation is applied')
Flags.DEFINE_boolean('random_crop', True, 'Whether perform the random crop')
Flags.DEFINE_integer('crop_size', 24, 'The crop size of the training image')
Flags.DEFINE_integer('name_queue_capacity', 2048, 'The capacity of the filename queue (suggest large to ensure'
                                                  'enough random shuffle.')
Flags.DEFINE_integer('image_queue_capacity', 2048, 'The capacity of the image queue (suggest large to ensure'
                                                   'enough random shuffle')
Flags.DEFINE_integer('queue_thread', 10, 'The threads of the queue (More threads can speedup the training process.')
# Generator configuration
Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')
# The content loss parameter
Flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
Flags.DEFINE_float('ratio', 0.001, 'The ratio between content loss and adversarial loss')
Flags.DEFINE_float('vgg_scaling', 0.0061, 'The scaling factor for the perceptual loss if using vgg perceptual loss')
# The training parameters
Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
Flags.DEFINE_integer('max_iter', 100000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 1000, 'The frequency of saving images')
Flags.DEFINE_integer('channel_num', 191, 'The number of images')
Flags.DEFINE_string('perceptual_mode', 'MSE', 'The type of feature used in perceptual loss')

FLAGS = Flags.FLAGS

# Print the configuration of the model
print_configuration_op(FLAGS)

# Check the output_dir is given
if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')

# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

# Check the summary directory to save the event
if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

# The testing mode
if FLAGS.mode == 'test':
    # Check the checkpoint
    if FLAGS.checkpoint is None:
        raise ValueError('The checkpoint file is needed to performing the test.')

    data_HR = scio.loadmat(r'./data/test.mat')
    data_LR = scio.loadmat(r'./data/test_ds.mat')
    HR = data_HR['test']
    LR = data_LR['test_ds']

    LR = np.expand_dims(LR, axis=4)
    OR = LR
    LR = np.transpose(LR, axes=[0, 3, 1, 2, 4])

    HR = HR.astype(np.float32)
    LR = LR.astype(np.float32)


    with tf.variable_scope('generator'):
        if FLAGS.task == '_3DSRGAN' or FLAGS.task == '_3DSRResnet':
            gen_output = generator(LR[[0]], 191, reuse=False, FLAGS=FLAGS)
        else:
            raise NotImplementedError('Unknown task!!')

    print('Finish building the network')

    # Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    weight_initiallizer = tf.train.Saver(var_list)

    # Define the initialization operation
    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Load the pretrained model
        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, FLAGS.checkpoint)

        result = sess.run(gen_output)
        scio.savemat('./result/result.mat', {'LR': gen_output.eval(), 'HR': HR[[0]], 'ori': OR[[0]]})


# The training mode
elif FLAGS.mode == 'train':

    inputs_batch, targets_batch = load_data(FLAGS)

    # Connect to the network
    if FLAGS.task == '_3DSRGAN':
        Net = _3DSRGAN(inputs_batch, targets_batch, FLAGS)
    elif FLAGS.task == '_3DSRResnet':
        Net = _3DSRResnet(inputs_batch, targets_batch, FLAGS)
    else:
        raise NotImplementedError('Unknown task type')

    print('Finish building the network!!!')

    # Compute PSNR
    with tf.name_scope("compute_psnr"):
        psnr = compute_psnr(targets_batch, Net.gen_output)

    # Add scalar summary
    if FLAGS.task == '_3DSRGAN':
        tf.summary.scalar('discriminator_loss', Net.discrim_loss)
        tf.summary.scalar('adversarial_loss', Net.adversarial_loss)
        tf.summary.scalar('content_loss', Net.content_loss)
        tf.summary.scalar('spectral_loss', Net.spectral_loss)
        tf.summary.scalar('generator_loss', Net.content_loss + Net.spectral_loss + FLAGS.ratio * Net.adversarial_loss)
        tf.summary.scalar('PSNR', psnr)
        tf.summary.scalar('learning_rate', Net.learning_rate)
        tf.summary.scalar('gen_loss', Net.gen_loss)
    elif FLAGS.task == '_3DSRResnet':
        # tf.summary.scalar('spectral_loss', Net.spectral_loss)
        tf.summary.scalar('content_loss', Net.content_loss)
        tf.summary.scalar('generator_loss', Net.content_loss)
        tf.summary.scalar('PSNR', psnr)
        tf.summary.scalar('learning_rate', Net.learning_rate)
        tf.summary.scalar('gen_loss', Net.gen_loss)

    # Define the saver and weight initiallizer
    saver = tf.train.Saver(max_to_keep=10)

    # The variable list
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    # Here if we restore the weight from the _3DSRResnet the var_list2 do not need to contain the discriminator weights
    # On contrary, if you initial your weight from other _3DSRGAN checkpoint, var_list2 need to contain discriminator
    # weights.
    if FLAGS.task == '_3DSRGAN':
        if FLAGS.pre_trained_model_type == '_3DSRGAN':
            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator') + \
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        elif FLAGS.pre_trained_model_type == '_3DSRResnet':
            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        else:
            raise ValueError('Unknown pre_trained model type!!')

    elif FLAGS.task == '_3DSRResnet':
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    weight_initiallizer = tf.train.Saver(var_list2)

    # Start the session
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True

    init_op = tf.global_variables_initializer()

    # Use superviser to coordinate all queue and summary writer

    sv = tf.train.Supervisor(logdir=FLAGS.summary_dir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        #use the pre_trained_model to continue the process of training
        if(FLAGS.pre_trained_model_type=='_3DSRGAN'):
            print('Loading model from the _3DSRGAN checkpoint...')
            weight_initiallizer.restore(sess, FLAGS.checkpoint)
        elif(FLAGS.pre_trained_model_type=='_3DSRResnet'):
            print('Loading model from the _3DSRResnet checkpoint...')
            weight_initiallizer.restore(sess, FLAGS.checkpoint)

        # Performing the training
        if FLAGS.max_epoch is None:
            if FLAGS.max_iter is None:
                raise ValueError('one of max_epoch or max_iter should be provided')
            else:
                max_iter = FLAGS.max_iter

        print('Optimization starts!!!')
        start = time.time()
        for step in range(max_iter):
            # 创建返回结果的fetches
            fetches = {
                "train": Net.train,
                "global_step": sv.global_step,
            }

            if ((step + 1) % FLAGS.display_freq) == 0:
                if FLAGS.task == '_3DSRGAN':
                    fetches["discrim_loss"] = Net.discrim_loss
                    fetches["adversarial_loss"] = Net.adversarial_loss
                    fetches["content_loss"] = Net.content_loss
                    fetches["spectral_loss"] = Net.spectral_loss
                    fetches["PSNR"] = psnr
                    fetches["learning_rate"] = Net.learning_rate
                    fetches["global_step"] = Net.global_step
                    fetches["gen_loss"] = Net.gen_loss

                elif FLAGS.task == '_3DSRResnet':
                    # fetches["spectral_loss"] = Net.spectral_loss
                    fetches["content_loss"] = Net.content_loss
                    fetches["PSNR"] = psnr
                    fetches["learning_rate"] = Net.learning_rate
                    fetches["global_step"] = Net.global_step
                    # fetches['spectral_loss'] = Net.spectral_loss
                    fetches['gen_loss'] = Net.gen_loss

            if ((step + 1) % FLAGS.summary_freq) == 0:
                fetches["summary"] = sv.summary_op
            # inputs_batch_train, targets_batch_train = sess.run([inputs_batch_shuffle, targets_batch_shuffle])
            results = sess.run(fetches)

            if ((step + 1) % FLAGS.summary_freq) == 0:
                print('Recording summary!!')
                sv.summary_writer.add_summary(results['summary'], results['global_step'])

            if ((step + 1) % FLAGS.display_freq) == 0:
                if FLAGS.task == '_3DSRGAN':
                    print("global_step", results["global_step"])
                    print("PSNR", results["PSNR"])
                    print("discrim_loss", results["discrim_loss"])
                    print("spectral_loss", results["spectral_loss"])
                    print("adversarial_loss", results["adversarial_loss"])
                    print("content_loss", results["content_loss"])
                    print("learning_rate", results['learning_rate'])
                    print("gen_loss", results['gen_loss'])
                elif FLAGS.task == '_3DSRResnet':
                    print("global_step", results["global_step"])
                    print("PSNR", results["PSNR"])
                    # print("spectral_loss", results["spectral_loss"])
                    print("content_loss", results["content_loss"])
                    print("learning_rate", results['learning_rate'])
                    print("gen_loss", results['gen_loss'])

            if ((step + 1) % FLAGS.save_freq) == 0:
                print('Save the checkpoint')
                saver.save(sess, os.path.join(FLAGS.output_dir, 'model'), global_step=sv.global_step)

        print('Optimization done!!!!!!!!!!!!')
