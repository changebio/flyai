# -*- coding: utf-8 -*

import numpy
from flyai.processor.base import Base
import cv2
from path import DATA_PATH
import os
import numpy as np

'''
把样例项目中的processor.py件复制过来替换即可
'''


class Processor(Base):
    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''
    def __init__(self):
        self.label_list = [2.0, 0.2, 10.0, 50.0, 0.5, 100.0, 0.1, 5.0, 1.0]
        self.img_shape = [224, 224, 3]

    def input_x(self, image_path):
        img = cv2.imread(os.path.join(DATA_PATH, image_path))
        img = cv2.resize(img, (self.img_shape[0], self.img_shape[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        return img

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''

    def input_y(self, label):
        index = self.label_list.index(label)
        # to onehot
        one_hot = np.zeros((len(self.label_list)))
        one_hot[index] = 1
        return one_hot

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def output_x(self, image_path):
        return image_path

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, label):
        return label
