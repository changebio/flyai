# -*- coding: utf-8 -*

import numpy
from data_helper import *
from flyai.processor.base import Base


MAX_LEN = 33


class Processor(Base):
    def __init__(self):
        super(Processor, self).__init__()
        self.word_dict = load_dict()

    def input_x(self, text):
        '''
        参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
        和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
        该方法字段与app.yaml中的input:->columns:对应
        '''

        sent_ids = word2id(text, self.word_dict, MAX_LEN)
        return sent_ids

    def input_y(self, label):
        '''
        参 数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
        和dataset.next_validation_batch()多次调用。
        该方法字段与app.yaml中的output:->columns:对应
        '''
        if label == 1:
            return [1, 0]
        else:
            return [0, 1]

    def output_y(self, data):
        '''
        输出的结果，会被dataset.to_categorys(data)调用
        '''
        if data[0] == 0:
            return 1
        else:
            return 0
