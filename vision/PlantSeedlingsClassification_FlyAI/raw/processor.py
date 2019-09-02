# -*- coding: utf-8 -*

import os
import numpy as np
import path as data_path
from flyai.processor.base import Base
from PIL import Image
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


'''
把样例项目中的processor.py件复制过来替换即可
'''


class Processor(Base):

    def __init__(self):
        # get a set of unique text labels
        self.list_labels = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
                            'Small-flowered Cranesbill', 'Fat Hen', 'Loose Silky-bent', 'Maize',
                            'Scentless Mayweed', 'Shepherds Purse', 'Sugar beet']
        # integer encode
        self.label_encoder = LabelEncoder()
        self.label_integer_encoded = self.label_encoder.fit_transform(self.list_labels)

        # one hot encode
        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.integer_encoded = self.label_integer_encoded.reshape(len(self.label_integer_encoded), 1)
        self.encoded_test = self.onehot_encoder.fit_transform(self.integer_encoded)
        self.inverted_test = argmax(self.encoded_test[0])

        # Map integer value to text labels
        self.label_to_int = {k: v for v, k in enumerate(self.list_labels)}

        # map integer to text labels
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}

    def convert_to_onehot(self, label):
        onehot = np.zeros((len(self.list_labels)))
        onehot[label] = 1
        return onehot

    def convert_to_label(self, onehot):
        return np.argmax(onehot)

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''
    def input_x(self, path):
        image = Image.open(os.path.join(data_path.DATA_PATH, path)).convert('L')
        image = image.crop((32, 32, 223, 223))
        image = image.resize((128, 128))
        x_data = np.array(image)
        x_data = x_data.astype(np.float32)
        x_data = x_data.reshape([128, 128, 1])
        x_data = np.multiply(x_data, 1.0 / 255.0)  ## scale to [0,1] from [0,255]
        x_data = np.transpose(x_data, (2, 0, 1))  ## reshape
        return x_data

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''

    def input_y(self, seedling):
        y_train_integer_encoded = self.label_to_int[seedling]
        y_train = self.convert_to_onehot(y_train_integer_encoded)
        return y_train

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def output_x(self, path):
        return path

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, data):
        label = self.convert_to_label(data)
        y = self.int_to_label[label]
        return y
