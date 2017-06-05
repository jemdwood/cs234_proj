#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train-action-prediction.py
# Author: Timon Ruban

import numpy as np
import os
import random
import argparse

import cv2
import tensorflow as tf
import six
from six.moves import queue

from tensorpack import *
from tensorpack.utils.serialize import *
from tensorpack.utils.stats import *
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient

from tensorpack.RL import *
from records_dataflow import RecordsDataFlow

if six.PY3:
    from concurrent import futures
    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

BATCH_SIZE = 128
PREDICT_BATCH_SIZE = 15     # batch for efficient forward
PREDICTOR_THREAD_PER_GPU = 3
PREDICTOR_THREAD = None

NUM_ACTIONS = 4


class Model(ModelDesc):
    def _get_inputs(self):
        assert NUM_ACTIONS is not None
        return [InputDesc(tf.uint8, (None,) + IMAGE_SHAPE3, 'state'),
                InputDesc(tf.int64, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'futurereward')]

    def _get_NN_prediction(self, image):
        image = tf.cast(image, tf.float32) / 255.0
        with argscope(Conv2D, nl=tf.nn.relu):
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

        l = FullyConnected('fc0', l, 512, nl=tf.identity)
        l = PReLU('prelu', l)
        logits = FullyConnected('fc-action', l, out_dim=NUM_ACTIONS, nl=tf.identity)    # unnormalized policy
        return logits

    def _build_graph(self, inputs):
        state, action, futurereward = inputs
        with tf.variable_scope('potential'):
            logits = self._get_NN_prediction(state)
        prob = tf.nn.softmax(logits, name='prob')   # a Bx4 with probabilities

        # a vector of length B with loss of each sample
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=action)
        self.cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss

        # compute the "incorrect vector", for the callback ClassificationError to use at validation time
        wrong = symbf.prediction_incorrect(logits, action, name='incorrect')
        accuracy = symbf.accuracy(logits, action, name='accuracy')

        # This will monitor training error (in a moving_average fashion):
        # 1. write the value to tensosrboard
        # 2. write the value to stat.json
        # 3. print the value after each epoch
        train_error = tf.reduce_mean(wrong, name='train_error')
        summary.add_moving_summary(train_error, accuracy)
        summary.add_moving_summary(self.cost)

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 0.001, summary=True)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)

        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


def get_data():
    rec_train = RecordsDataFlow('train')
    rec_test = RecordsDataFlow('test')
    return BatchData(rec_train, BATCH_SIZE), BatchData(rec_test, BATCH_SIZE, remainder=True)

def get_config():
    dirname = os.path.join('train_log', 'action_prediction')
    logger.set_logger_dir(dirname)
    M = Model()

    dataset_train, dataset_test = get_data()
    # How many iterations you want in each epoch.
    # This is the default value, don't actually need to set it in the config
    steps_per_epoch = dataset_train.size()

    # get the config which contains everything necessary in a training
    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,  # the DataFlow instance for training
        callbacks=[
            ModelSaver(),   # save the model after every epoch
            ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            MaxSaver('validation_accuracy'),  # save the model with highest accuracy (prefix 'validation_')
            InferenceRunner(    # run inference(for validation) after every epoch
                dataset_test,   # the DataFlow instance used for validation
                # Calculate both the cost and the error for this DataFlow
                [ScalarStats('cross_entropy_loss'), ScalarStats('accuracy'),
                 ClassificationError('incorrect')]),
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=100,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',
                        choices=['eval', 'train'], default='train')
    parser.add_argument('--episode', help='number of episode to eval', default=100, type=int)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.task != 'train':
        assert args.load is not None

    if args.task != 'train':
        print('Not implemented yet')
    else:
        nr_gpu = get_nr_gpu()
        if nr_gpu > 0:
            if nr_gpu > 1:
                predict_tower = list(range(nr_gpu))[-nr_gpu // 2:]
            else:
                predict_tower = [0]
            PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
            train_tower = list(range(nr_gpu))[:-nr_gpu // 2] or [0]
            logger.info("[BA3C] Train on gpu {} and infer on gpu {}".format(
                ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
            trainer = AsyncMultiGPUTrainer
        else:
            logger.warn("Without GPU this model will never learn! CPU is only useful for debug.")
            nr_gpu = 0
            PREDICTOR_THREAD = 1
            predict_tower, train_tower = [0], [0]
            trainer = QueueInputTrainer
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        config.tower = train_tower
        config.predict_tower = predict_tower
        trainer(config).train()
