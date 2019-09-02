# -*- coding: utf-8 -*
import time

import argparse
import numpy as np
import os
import tensorflow as tf
from flyai.dataset import Dataset

import conf
import net
from model import Model
from path import MODEL_PATH

input_seq_len = conf.input_seq_len
output_seq_len = conf.output_seq_len
# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=5, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
args = parser.parse_args()
# 数据获取辅助类

dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

# ——————————————————训练模型——————————————————

"""
训练过程
"""
with tf.Session() as sess:
    encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = net.get_model()

    sess.run(tf.global_variables_initializer())

    # 训练很多次迭代，每隔10次打印一次loss，可以看情况直接ctrl+c停止
    for i in range(dataset.get_step()):
        x_train, y_train = dataset.next_train_batch()
        sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = net.processor.get_samples(
            [x_train, y_train], input_seq_len, output_seq_len)
        input_feed = {}
        for l in range(input_seq_len):
            input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
        for l in range(output_seq_len):
            input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
            input_feed[target_weights[l].name] = sample_target_weights[l]
        input_feed[decoder_inputs[output_seq_len].name] = np.zeros([len(sample_decoder_inputs[0])], dtype=np.int32)
        [loss_ret, _] = sess.run([loss, update], input_feed)
        if i % 100 == 0:
            print(time.ctime(), 'step=', i, 'loss=', loss_ret, 'learning_rate=', learning_rate.eval())
            # 模型持久化
            model.save_model(saver, sess, os.path.join(MODEL_PATH, ''), overwrite=True)
        if i % 100 == 0:
            sess.run(learning_rate_decay_op)
