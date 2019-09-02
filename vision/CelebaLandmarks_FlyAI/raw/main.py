import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

# 数据获取辅助类
dataset = Dataset()
# 模型操作辅助类
model = Model(dataset)

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()


def vgg16(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.conv2d(inputs, 64, [3, 3], scope='conv1')

        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        net = slim.conv2d(net, 100, [3, 3], scope='conv4')

        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        net = slim.conv2d(net, 256, [3, 3], scope='conv5')

        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        net = slim.fully_connected(net, 128, scope='fc7')

        net = slim.dropout(net, scope='dropout7')

        net = slim.fully_connected(net, 60, activation_fn=None, scope='fc8')

    return net


def fc(name, x, out_channel, keep_prob=1.0):
    x = tf.layers.batch_normalization(x, training=True)
    shape = x.get_shape().as_list()
    if len(shape) == 4:
        size = shape[-1] * shape[-2] * shape[-3]
    else:
        size = shape[1]
    x_flat = tf.reshape(x, [-1, size])
    with tf.variable_scope(name):
        W = tf.get_variable(name='weights', shape=[size, out_channel], dtype=tf.float32)
        b = tf.get_variable(name='biases', shape=[out_channel], initializer=tf.constant_initializer(100.0),
                            dtype=tf.float32)

        res = tf.matmul(x_flat, W)
        res = tf.nn.dropout(res, keep_prob)
        out = tf.nn.relu(tf.nn.bias_add(res, b))

    return out


input_x = tf.placeholder(tf.float32, shape=[None, 178, 218, 3], name="input_x")

input_y = tf.placeholder(tf.float32, shape=[None, 10], name="input_y")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
y_ = vgg16(input_x)

y_conv = fc(name="fc", x=y_, out_channel=10)  #
y_conv = tf.add(y_conv, 0, name="y_conv")

loss = slim.losses.absolute_difference(input_y, y_conv)
train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(args.EPOCHS):
        x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)
        sess.run(train_step, feed_dict={input_x: x_train, input_y: y_train, keep_prob: 0.5})
        train_acc = sess.run(loss, feed_dict={input_x: x_train, input_y: y_train, keep_prob: 1.0})
        model.save_model(sess, MODEL_PATH, overwrite=True)
        print('epoch: ', str(i + 1) + "/" + str(args.EPOCHS), 'loss:', train_acc)

print("*******testing*******",x_train,y_train)
import random
_,_,x_test,y_text = dataset.get_all_data()
test_idx = random.sample(range(len(x_test)),10)
print(model.predict_all([x_test[i] for i in test_idx]),[y_text[i] for i in test_idx])
