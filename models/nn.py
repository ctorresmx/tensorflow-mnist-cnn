import tensorflow as tf

from models.basic_model import BasicModel


class NeuralNetwork(BasicModel):
    def infer(self, input, train):
        layer_1 = tf.layers.dense(inputs=input, units=350,
                                  activation=tf.nn.relu)

        layer_2 = tf.layers.dense(inputs=layer_1, units=50,
                                  activation=tf.nn.relu)

        y = tf.layers.dense(inputs=layer_2, units=10,
                            activation=tf.nn.softmax, name='predict')

        return y

    def cost(self, logits, labels):
        return tf.reduce_mean(-tf.reduce_sum(
            labels * tf.log(logits), axis=[1])
        )

    def optimize(self, cost, learning_rate):
        return tf.train.AdamOptimizer().minimize(cost)

    def evaluate(self, logits, labels):
        correct_prediction = tf.equal(tf.argmax(logits, 1),
                                      tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy
