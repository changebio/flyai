# -*- coding: utf-8 -*

import sys

import jieba
import json
import numpy as np
import os
from flyai.processor.base import Base

import conf
from path import DATA_PATH


class Processor(Base):
    def __init__(self):
        with open(os.path.join(DATA_PATH, 'words.dict'), 'r') as f:
            js = f.read()
            self.word2id_dict = json.loads(js)
        self.id2word_dict = {}
        for k, v in self.word2id_dict.items():
            self.id2word_dict[v] = k
        self.length = len(self.word2id_dict) + 5
        # 空值填充0
        self.PAD_ID = 0
        # 输出序列起始标记
        self.GO_ID = 1
        # 结尾标记
        self.EOS_ID = 2
        self.UNK = 3
        self.input_seq_len = conf.input_seq_len
        self.output_seq_len = conf.output_seq_len

    def input_x(self, question):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        return self.get_id_list_from(question)

    def input_y(self, answer):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        return self.get_id_list_from(answer)

    def output_y(self, outputs_seq):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        outputs_seq = [self.id2word(v) for v in outputs_seq]

        return "".join(outputs_seq)

    def id2word(self, id):
        id = int(id)
        if id in self.id2word_dict:
            return self.id2word_dict[id]
        else:
            return ""

    def get_id_list_from(self, sentence):
        sentence_id_list = []
        seg_list = jieba.cut(sentence)
        for str in seg_list:
            sentence_id_list.append(self.word2id(str))
        return sentence_id_list

    def word2id(self, word):
        if not isinstance(word, str):
            print("Exception: error word not unicode")
            sys.exit(1)
        if word in self.word2id_dict:
            return self.word2id_dict[word]
        else:
            return 3  # UNK

    def get_samples(self, train_set, input_seq_len, output_seq_len):
        """构造样本数据

        :return:
            encoder_inputs: [array([0, 0], dtype=int32), array([0, 0], dtype=int32), array([5, 5], dtype=int32),
                            array([7, 7], dtype=int32), array([9, 9], dtype=int32)]
            decoder_inputs: [array([1, 1], dtype=int32), array([11, 11], dtype=int32), array([13, 13], dtype=int32),
                            array([15, 15], dtype=int32), array([2, 2], dtype=int32)]
        """

        raw_encoder_input = []
        raw_decoder_input = []

        for i in range(len(train_set[0])):
            raw_encoder_input.append([self.PAD_ID] * (input_seq_len - len(train_set[0][i])) + list(train_set[0][i]))
            raw_decoder_input.append(
                [self.GO_ID] + list(train_set[1][i]) + [self.PAD_ID] * (output_seq_len - len(train_set[1][i]) - 1))

        encoder_inputs = []
        decoder_inputs = []
        target_weights = []

        for length_idx in range(input_seq_len):
            encoder_inputs.append(
                np.array([encoder_input[length_idx] for encoder_input in raw_encoder_input], dtype=np.int32))
        for length_idx in range(output_seq_len):
            decoder_inputs.append(
                np.array([decoder_input[length_idx] for decoder_input in raw_decoder_input], dtype=np.int32))
            target_weights.append(np.array([
                0.0 if length_idx == output_seq_len - 1 or decoder_input[length_idx] == self.PAD_ID else 1.0 for
                decoder_input in raw_decoder_input
            ], dtype=np.float32))
        return encoder_inputs, decoder_inputs, target_weights

    def seq_to_encoder(self, input_seq):
        """从输入空格分隔的数字id串，转成预测用的encoder、decoder、target_weight等
        """
        input_seq_array = [int(v) for v in input_seq.split()]
        encoder_input = [self.PAD_ID] * (self.input_seq_len - len(input_seq_array)) + input_seq_array
        decoder_input = [self.GO_ID] + [self.PAD_ID] * (self.output_seq_len - 1)
        encoder_inputs = [np.array([v], dtype=np.int32) for v in encoder_input]
        decoder_inputs = [np.array([v], dtype=np.int32) for v in decoder_input]
        target_weights = [np.array([1.0], dtype=np.float32)] * self.output_seq_len
        return encoder_inputs, decoder_inputs, target_weights
