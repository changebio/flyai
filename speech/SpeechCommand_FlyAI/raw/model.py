# -*- coding: utf-8 -*
import numpy as np
import os
import tensorflow as tf
from flyai.model.base import Base
from tensorflow.python.saved_model import tag_constants

import create_dict
from path import MODEL_PATH

TENSORFLOW_MODEL_DIR = "best"

time_shift_ms = 100.0
sample_rate = 16000
clip_duration_ms = 1000
window_size_ms = 30.0
window_stride_ms = 10.0
dct_coefficient_count = 40
label_count = 12


def prepare_model_settings():
    """Calculates common settings needed for all models.

    Args:
      label_count: How many classes are to be recognized.
      sample_rate: Number of audio samples per second.
      clip_duration_ms: Length of each audio clip to be analyzed.
      window_size_ms: Duration of frequency analysis window.
      window_stride_ms: How far to move in time between frequency windows.
      dct_coefficient_count: Number of frequency bins to use for analysis.

    Returns:
      Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'dct_coefficient_count': dct_coefficient_count,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
    }


def create_conv_model(fingerprint_input, model_settings, is_training):
    """Builds a standard convolutional model.

    This is roughly the network labeled as 'cnn-trad-fpool3' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

    Here's the layout of the graph:

    (fingerprint_input)
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MaxPool]
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MaxPool]
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v

    This produces fairly good quality results, but can involve a large number of
    weight parameters and computations. For a cheaper alternative from the same
    paper with slightly less accuracy, see 'low_latency_conv' below.

    During training, dropout nodes are introduced after each relu, controlled by a
    placeholder.

    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    if is_training:
        dropout_prob = tf.placeholder(tf.float32, name='keep_prob')
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 8
    first_filter_height = 20
    first_filter_count = 64
    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))
    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1, 1, 1, 1],
                              'SAME') + first_bias
    first_relu = tf.nn.relu(first_conv)
    if is_training:
        first_dropout = tf.nn.dropout(first_relu, dropout_prob)
    else:
        first_dropout = first_relu
    max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    second_filter_width = 4
    second_filter_height = 10
    second_filter_count = 64
    second_weights = tf.Variable(
        tf.truncated_normal(
            [
                second_filter_height, second_filter_width, first_filter_count,
                second_filter_count
            ],
            stddev=0.01))
    second_bias = tf.Variable(tf.zeros([second_filter_count]))
    second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1],
                               'SAME') + second_bias
    second_relu = tf.nn.relu(second_conv)
    if is_training:
        second_dropout = tf.nn.dropout(second_relu, dropout_prob)
    else:
        second_dropout = second_relu
    second_conv_shape = second_dropout.get_shape()
    second_conv_output_width = second_conv_shape[2]
    second_conv_output_height = second_conv_shape[1]
    second_conv_element_count = int(
        second_conv_output_width * second_conv_output_height *
        second_filter_count)
    flattened_second_conv = tf.reshape(second_dropout,
                                       [-1, second_conv_element_count])
    label_count = model_settings['label_count']
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [second_conv_element_count, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
    if is_training:
        return final_fc, dropout_prob
    else:
        return final_fc


class CNN_Model(object):
    def __init__(self):
        model_settings = prepare_model_settings()
        fingerprint_size = model_settings['fingerprint_size']
        label_count = model_settings['label_count']

        fingerprint_input = tf.placeholder(
            tf.float32, [None, fingerprint_size], name='input_x')

        logits, dropout_prob = create_conv_model(
            fingerprint_input,
            model_settings,
            is_training=True)

        ground_truth_input = tf.placeholder(
            tf.float32, [None, label_count], name='y_input')
        predicted_indices = tf.argmax(logits, 1)
        expected_indices = tf.argmax(ground_truth_input, 1)
        correct_prediction = tf.equal(predicted_indices, expected_indices)
        self.confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices)
        self.evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.prob = tf.nn.softmax(logits, name='y_conv')
        # Define loss and optimizer
        self.fingerprint_input = fingerprint_input
        self.model_settings = model_settings
        self.logits = logits
        self.dropout_prob = dropout_prob
        self.ground_truth_input = ground_truth_input

    def train_model(self, sess, dataset, model_helper, batch_size, epochs):
        logits = self.logits
        model_settings = self.model_settings
        dropout_prob = self.dropout_prob
        fingerprint_input = self.fingerprint_input
        ground_truth_input = self.ground_truth_input
        evaluation_step = self.evaluation_step
        confusion_matrix = self.confusion_matrix
        eval_step_interval = 400

        control_dependencies = []
        # Create the back propagation and training evaluation machinery in the graph.
        with tf.name_scope('cross_entropy'):
            cross_entropy_mean = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=ground_truth_input, logits=logits))
        with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
            learning_rate_input = tf.placeholder(
                tf.float32, [], name='learning_rate_input')
            train_step = tf.train.AdamOptimizer(
                learning_rate_input, epsilon=1e-6).minimize(cross_entropy_mean)

        global_step = tf.contrib.framework.get_or_create_global_step()
        increment_global_step = tf.assign(global_step, global_step + 1)

        tf.global_variables_initializer().run()

        params = [param for param in tf.global_variables() if
                  ('dense_fn' not in param.name and 'global_step' not in param.name)]
        start_step = global_step.eval(session=sess)

        print('Training from step: %d ', start_step)
        best = 0
        # Training loop.
        for training_step in range(start_step, epochs):
            # Figure out what the current learning rate is.
            # Pull the audio samples we'll use for training.
            train_fingerprints, train_ground_truth, _, _ = dataset.next_batch(batch_size, test_data=False)
            # Run the graph with this batch of training data.
            train_accuracy, cross_entropy_value, _, _ = sess.run(
                [
                    evaluation_step, cross_entropy_mean, train_step,
                    increment_global_step
                ],
                feed_dict={
                    fingerprint_input: train_fingerprints,
                    ground_truth_input: train_ground_truth,
                    learning_rate_input: 0.001,
                    dropout_prob: 1.0
                })

            # train_writer.add_summary(train_summary, training_step)
            # if (training_step % FLAGS.eval_step_interval) == 0:
            print('Step #%d: accuracy %.1f%%, cross entropy %f' %
                  (training_step, train_accuracy * 100,
                   cross_entropy_value))
            if train_accuracy > best:
                model_helper.save_model(sess, MODEL_PATH, overwrite=True)
                best = train_accuracy
                print("save model")

    def evaluate(self, sess, x_test, y_test, batch_size=32):
        logits = self.logits
        model_settings = self.model_settings
        dropout_prob = self.dropout_prob
        fingerprint_input = self.fingerprint_input
        ground_truth_input = self.ground_truth_input
        evaluation_step = self.evaluation_step
        confusion_matrix = self.confusion_matrix
        set_size = x_test.shape[0]
        total_accuracy = 0
        total_cross_entropy = 0.0
        total_conf_matrix = None
        for i in range(0, set_size, batch_size):
            validation_fingerprints, validation_ground_truth = x_test[i:i + batch_size], \
                                                               y_test[i:i + batch_size]
            # Run a validation step and capture training summaries for TensorBoard
            # with the `merged` op.
            validation_accuracy = sess.run(
                evaluation_step,
                feed_dict={
                    fingerprint_input: validation_fingerprints,
                    ground_truth_input: validation_ground_truth,
                    dropout_prob: 1.0,
                })
            # validation_writer.add_summary(validation_summary, training_step)
            tmp_batch_size = min(batch_size, set_size - i)
            total_accuracy += (validation_accuracy * tmp_batch_size) / set_size
        return total_accuracy

    def predict(self, sess, x):
        dropout_prob = self.dropout_prob
        fingerprint_input = self.fingerprint_input
        test_prob = sess.run(self.prob, feed_dict={fingerprint_input: x,
                                                   dropout_prob: 1.0
                                                   })
        return test_prob


class Model(Base):
    def __init__(self, data, do_train=False):
        self.data = data
        self.label_dict, self.label_dict_res = create_dict.load_label_dict()
        self.num_tags = max(self.label_dict.values()) + 1
        if do_train:
            self.model = CNN_Model()

    def predict(self, **data):
        '''
        使用模型
        :param path: 模型所在的路径
        :param name: 模型的名字
        :param data: 模型的输入参数
        :return:
        '''

        with tf.Session() as sess:
            tf.saved_model.loader.load(sess, [tag_constants.SERVING], os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR))
            input_x = sess.graph.get_tensor_by_name(self.get_tensor_name('input_x'))
            y_conv = sess.graph.get_tensor_by_name(self.get_tensor_name('y_conv'))
            keep_prob = sess.graph.get_tensor_by_name(self.get_tensor_name('keep_prob'))
            x_data = self.data.predict_data(**data)
            prob = sess.run(y_conv, feed_dict={input_x: x_data, keep_prob: 1.0})
            y = np.argmax(np.squeeze(prob, axis=0))
            return y

    def predict_all(self, datas):
        with tf.Session() as sess:
            tf.saved_model.loader.load(sess, [tag_constants.SERVING], os.path.join(MODEL_PATH, TENSORFLOW_MODEL_DIR))
            input_x = sess.graph.get_tensor_by_name(self.get_tensor_name('input_x'))
            y_conv = sess.graph.get_tensor_by_name(self.get_tensor_name('y_conv'))
            keep_prob = sess.graph.get_tensor_by_name(self.get_tensor_name('keep_prob'))
            labels = []
            for data in datas:
                x_data = self.data.predict_data(**data)
                prob = sess.run(y_conv, feed_dict={input_x: x_data, keep_prob: 1.0})
                y = np.argmax(np.squeeze(prob, axis=0))
                labels.append(self.data.to_categorys(y))
            return labels

    def save_model(self, session, path, name=TENSORFLOW_MODEL_DIR, overwrite=False):
        '''
        保存模型
        :param session: 训练模型的sessopm
        :param path: 要保存模型的路径
        :param name: 要保存模型的名字
        :param overwrite: 是否覆盖当前模型
        :return:
        '''
        if overwrite:
            self.delete_file(path)

        builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(path, name))
        builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.SERVING])
        builder.save()

    def evaluate(self, x_test, y_test, path, name=TENSORFLOW_MODEL_DIR):
        '''
        验证模型
        :param path: 模型的路径
        :param name: 模型的名字
        :return: 返回验证的准确率
        '''
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(os.path.join(path, name))
            checkpoint_suffix = ""
            if tf.__version__ > "0.12":
                checkpoint_suffix = ".index"
            if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                return None
            score = self.model.evaluate(sess, x_test, y_test)
            return score

    def get_tensor_name(self, name):
        return name + ":0"

    def delete_file(self, path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
