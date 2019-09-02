# -*- coding: utf-8 -*
import numpy
from flyai.processor.base import Base


class Processor(Base):

    def input_x(self, text):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        return text

    def input_y(self, stars):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        one_hot_label = numpy.zeros([5])  ##生成全0矩阵
        one_hot_label[stars - 1] = 1  ##相应标签位置置
        return one_hot_label

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        return numpy.argmax(data)
