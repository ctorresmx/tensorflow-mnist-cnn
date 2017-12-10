#!/usr/bin/env python
# -*- coding: utf-8 -*-
import fire
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from models.cnn import ConvolutionalNeuralNetwork
from models.nn import NeuralNetwork
from models.perceptron import Perceptron

from utilities.logger import logger

class ModelManager:
    def __init__(self, batch_size=100, epochs=5, log_dir='results/',
                 model_to_use='perceptron'):
        logger.info('Loading MNIST data...')
        self._mnist_data = input_data.read_data_sets('data/', one_hot=True)
        self._batch_size = batch_size
        self._epochs = epochs
        self._log_dir = log_dir
        self._model_to_use = model_to_use

        self._models = {
            'cnn': ConvolutionalNeuralNetwork,
            'nn': NeuralNetwork,
            'perceptron': Perceptron,
        }

    def train(self):
        with tf.Session() as session:
            input = tf.placeholder(tf.float32, shape=[None, 784], name='input')
            train = tf.placeholder(tf.bool)
            labels = tf.placeholder(tf.float32, shape=[None, 10])

            logger.info('Initializing {} model...'.format(self._model_to_use))
            model = self._models[self._model_to_use]()

            logger.info('Preparing model...')
            inference = model.infer(input, train)
            cost = model.cost(inference, labels)
            optimize = model.optimize(cost, 0.1)
            evaluate = model.evaluate(inference, labels)

            logger.info('Initializing variables...')
            session.run(tf.global_variables_initializer())
            tf.summary.scalar('cost', cost)
            writer = tf.summary.FileWriter(self._log_dir, graph=tf.get_default_graph())
            summarize = tf.summary.merge_all()

            total_batch = int(self._mnist_data.train.num_examples / self._batch_size)
            step = 0
            for epoch in range(self._epochs):
                for _ in range(total_batch):
                    batch_x, batch_y = self._mnist_data.train.next_batch(self._batch_size)

                    _, cost_value, summary = session.run([optimize, cost, summarize], feed_dict={
                        input: batch_x, labels: batch_y, train: True,
                    })

                    step += 1
                    if step % 100 == 0:
                        logger.debug('epoch: {}, step: {}, cost: {:.2f}'.format(
                            epoch, step, cost_value)
                        )
                        writer.add_summary(summary, step)

            accuracy = session.run(evaluate, feed_dict={
                input: self._mnist_data.test.images,
                labels: self._mnist_data.test.labels,
                train: False,
            })

            logger.info('The accuracy is: {:.2f}%'.format(accuracy * 100))


if __name__ == '__main__':
    fire.Fire(ModelManager)
