import fire
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from models.perceptron import Perceptron


BATCH_SIZE = 100
EPOCHS = 10
LOG_DIR = 'results/'


class ModelManager:
    def __init__(self):
        self.mnist_data = input_data.read_data_sets('data/', one_hot=True)

    def train(self):
        with tf.Session() as session:
            input = tf.placeholder(tf.float32, shape=[None, 784], name='input')
            labels = tf.placeholder(tf.float32, shape=[None, 10])
            model = Perceptron()

            inference = model.infer(input)
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
                        input: batch_x, labels: batch_y
                    })

                    step += 1
                    if step % 100 == 0:
                        print('epoch: {}, step: {}, cost: {:.2f}'.format(
                            epoch, step, cost_value)
                        )
                        writer.add_summary(summary, step)

            accuracy = session.run(evaluate, feed_dict={
                input: self.mnist_data.test.images,
                labels: self.mnist_data.test.labels
            })

            print('The accuracy is: {:.2f}%'.format(accuracy * 100))


if __name__ == '__main__':
    fire.Fire(ModelManager)
