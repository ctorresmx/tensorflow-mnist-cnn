import fire
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from models.perceptron import Perceptron
from models.nn import NeuralNetwork
from models.cnn import ConvolutionalNeuralNetwork


BATCH_SIZE = 100
EPOCHS = 5
LOG_DIR = 'results/'


class ModelManager:
    def __init__(self):
        self.mnist_data = input_data.read_data_sets('data/', one_hot=True)

        self.models = {
            'perceptron': Perceptron,
            'nn': NeuralNetwork,
            'cnn': ConvolutionalNeuralNetwork,
        }

    def train(self, model_to_use='perceptron'):
        with tf.Session() as session:
            input = tf.placeholder(tf.float32, shape=[None, 784], name='input')
            train = tf.placeholder(tf.bool)
            labels = tf.placeholder(tf.float32, shape=[None, 10])
            model = self.models[model_to_use]()

            inference = model.infer(input, train)
            cost = model.cost(inference, labels)
            optimize = model.optimize(cost, 0.1)
            evaluate = model.evaluate(inference, labels)

            session.run(tf.global_variables_initializer())
            tf.summary.scalar('cost', cost)
            writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
            summarize = tf.summary.merge_all()

            total_batch = int(self.mnist_data.train.num_examples / BATCH_SIZE)
            step = 0
            for epoch in range(EPOCHS):
                for _ in range(total_batch):
                    batch_x, batch_y = self.mnist_data.train.next_batch(BATCH_SIZE)

                    _, cost_value, summary = session.run([optimize, cost, summarize], feed_dict={
                        input: batch_x, labels: batch_y, train: True,
                    })

                    step += 1
                    if step % 100 == 0:
                        print('epoch: {}, step: {}, cost: {:.2f}'.format(
                            epoch, step, cost_value)
                        )
                        writer.add_summary(summary, step)

            accuracy = session.run(evaluate, feed_dict={
                input: self.mnist_data.test.images,
                labels: self.mnist_data.test.labels,
                train: False,
            })

            print('The accuracy is: {:.2f}%'.format(accuracy * 100))


if __name__ == '__main__':
    fire.Fire(ModelManager)
