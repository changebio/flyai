"""
Derived from: https://www.cs.toronto.edu/~frossard/vgg16/vgg16.py
"""
import tensorflow as tf
import numpy as np


class VggNetModel(object):

    def __init__(self, num_classes=11):
        # ignored regions(0), pedestrian(1), people(2), bicycle(3), car(4),van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10)
        self.num_classes = num_classes

    def inference(self, x):
        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out, name=scope.name)

        # conv1_2
        with tf.variable_scope('conv1_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[64], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out, name=scope.name)

        # pool1
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out, name=scope.name)

        # conv2_2
        with tf.variable_scope('conv2_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[128], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out, name=scope.name)

        # pool2
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out, name=scope.name)

        # conv3_2
        with tf.variable_scope('conv3_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out, name=scope.name)

        # conv3_3
        with tf.variable_scope('conv3_3') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[256], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out, name=scope.name)

        # pool3
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out, name=scope.name)

        # conv4_2
        with tf.variable_scope('conv4_2') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out, name=scope.name)

        # conv4_3
        with tf.variable_scope('conv4_3') as scope:
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[512], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out, name=scope.name)

        # pool4
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        # conv_final
        with tf.variable_scope('conv_final') as scope:
            # self.num_classes + 4 其中 4 为 方框回归的4个值
            kernel = tf.get_variable('weights', initializer=tf.truncated_normal([1, 1, 512, self.num_classes+4], dtype=tf.float32, stddev=1e-1))
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', initializer=tf.constant(0.0, shape=[self.num_classes+4], dtype=tf.float32))
            out = tf.nn.bias_add(conv, biases)
            softmax_out = tf.nn.softmax(out[:,:,:,:-4])
            self.conv_final = tf.concat([softmax_out, out[:,:,:,-4:]], axis=-1) # [batch_size, 32, 32, 15]
        return out # 1/16

    # 重新定义
    def loss(self, batch_x, batch_y):
        y_predict = self.inference(batch_x)
        pred_label = y_predict[:,:,:,:-4]
        pred_bb = y_predict[:,:,:,-4:]
        batch_label = batch_y[:,:,:,:-4]
        batch_bb = batch_y[:,:,:,-4:]
        self.loss_label = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_label, labels=batch_label))

        self.loss_rmse = tf.sqrt(tf.losses.mean_squared_error(batch_bb, pred_bb))
        self.loss = self.loss_label + self.loss_rmse
        return self.loss

    def optimize(self, learning_rate):
        return tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

