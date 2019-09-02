# -*- coding: utf-8 -*
import numpy
import numpy as np
import os
import tensorflow as tf
from flyai.model.base import Base

import conf
import net
from path import MODEL_PATH

TENSORFLOW_MODEL_DIR = "best"


class Model(Base):
    def __init__(self, data):
        self.data = data
        self.input_seq_len = conf.input_seq_len
        self.output_seq_len = conf.output_seq_len
        self.EOS_ID = 2
        self.processor = net.processor

    def predict(self, **data):
        '''
        使用模型
        :param path: 模型所在的路径
        :param name: 模型的名字
        :param data: 模型的输入参数
        :return:
        '''
        with tf.Session() as sess:
            encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = net.get_model(
                feed_previous=True)
            ckpt = tf.train.get_checkpoint_state(os.path.join(MODEL_PATH, ''))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("No checkpoint file found")
                return
            input_id_list = self.data.predict_data(**data)[0]
            sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = self.processor.seq_to_encoder(
                ' '.join([str(v) for v in input_id_list]))
            input_feed = {}
            for l in range(self.input_seq_len):
                input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
            for l in range(self.output_seq_len):
                input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                input_feed[target_weights[l].name] = sample_target_weights[l]
            input_feed[decoder_inputs[self.output_seq_len].name] = np.zeros([2], dtype=np.int32)

            # 预测输出
            outputs_seq = sess.run(outputs, input_feed)
            # 因为输出数据每一个是num_decoder_symbols维的，因此找到数值最大的那个就是预测的id，就是这里的argmax函数的功能
            outputs_seq = [int(np.argmax(logit[0], axis=0)) for logit in outputs_seq]
            # 如果是结尾符，那么后面的语句就不输出了
            if self.EOS_ID in outputs_seq:
                outputs_seq = outputs_seq[:outputs_seq.index(self.EOS_ID)]
            return self.data.to_categorys(outputs_seq)

    def predict_all(self, datas):

        with tf.Session() as sess:
            encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = net.get_model(
                feed_previous=True)

            ckpt = tf.train.get_checkpoint_state(os.path.join(MODEL_PATH, ''))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("No checkpoint file found")
                return
            outputa = []
            for data in datas:
                input_id_list = self.data.predict_data(**data)[0]
                sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = self.processor.seq_to_encoder(
                    ' '.join([str(v) for v in input_id_list]))
                input_feed = {}
                for l in range(self.input_seq_len):
                    input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
                for l in range(self.output_seq_len):
                    input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                    input_feed[target_weights[l].name] = sample_target_weights[l]
                input_feed[decoder_inputs[self.output_seq_len].name] = np.zeros([2], dtype=np.int32)

                # 预测输出
                outputs_seq = sess.run(outputs, input_feed)
                # 因为输出数据每一个是num_decoder_symbols维的，因此找到数值最大的那个就是预测的id，就是这里的argmax函数的功能
                outputs_seq = [int(np.argmax(logit[0], axis=0)) for logit in outputs_seq]
                # 如果是结尾符，那么后面的语句就不输出了
                if self.EOS_ID in outputs_seq:
                    outputs_seq = outputs_seq[:outputs_seq.index(self.EOS_ID)]
                outputa.append({'answer': self.data.to_categorys(outputs_seq)})
        return outputa

    def save_model(self, saver, session, path, name=TENSORFLOW_MODEL_DIR, overwrite=False):
        '''
        保存模型
        :param session: 训练模型的sessopm
        :param path: 要保存模型的路径
        :param name: 要保存模型的名字
        :param overwrite: 是否覆盖当前模型
        :return:
        '''
        saver.save(session, path)

    def batch_iter(self, x, y, batch_size=128):
        '''
        生成批次数据
        :param x: 所有验证数据x
        :param y: 所有验证数据y
        :param batch_size: 每批的大小
        :return: 返回分好批次的数据
        '''
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def get_tensor_name(self, name):
        return name + ":0"

    def delete_file(self, path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
