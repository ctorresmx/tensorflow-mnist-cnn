import tensorflow as tf

from models.basic_model import BasicModel


class ConvolutionalNeuralNetwork(BasicModel):
    def infer(self, input, train):
        input_as_grid = tf.reshape(input, [-1, 28, 28, 1])

        conv_1 = tf.layers.conv2d(input_as_grid, filters=32, kernel_size=5,
                                  padding='same', activation=tf.nn.relu)
        pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=2, strides=2,
                                         padding='same')

        conv_2 = tf.layers.conv2d(pool_1, filters=64, kernel_size=5,
                                  padding='same', activation=tf.nn.relu)
        pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=2, strides=2,
                                         padding='same')

        flatten_tensor = tf.layers.flatten(conv_2)

        fc_1 = tf.layers.dense(inputs=flatten_tensor, units=512,
                               activation=tf.nn.relu)
        drop_1 = tf.layers.dropout(fc_1, training=train)

        fc_2 = tf.layers.dense(inputs=drop_1, units=10,
                               activation=tf.nn.softmax, name='predict')

        return fc_2

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
