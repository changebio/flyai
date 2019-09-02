# -*- coding: utf-8 -*

import numpy
import os
import librosa
import path
import numpy as np

from flyai.processor.base import Base
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
from keras.utils import to_categorical

'''
把样例项目中的processor.py件复制过来替换即可
'''


class Processor(Base):

    def __init__(self):
        # get a set of unique text labels
        self.list_labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
                            'drilling',
                            'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

        # integer encode
        self.label_encoder = LabelEncoder()
        self.label_integer_encoded = self.label_encoder.fit_transform(self.list_labels)

        # one hot encode
        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.integer_encoded = self.label_integer_encoded.reshape(len(self.label_integer_encoded),
                                                                  1)
        self.encoded_test = self.onehot_encoder.fit_transform(self.integer_encoded)
        self.inverted_test = argmax(self.encoded_test[0])

        # Map integer value to text labels
        self.label_to_int = {k: v for v, k in enumerate(self.list_labels)}
        # print ("test label to int ",label_to_int["Applause"])

        # map integer to text labels
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def input_x(self, ID):
        duration = 2.97
        sr = 22050
        y, sr = librosa.load(os.path.join(path.DATA_PATH, ID), duration=duration, sr=sr)
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        dur = librosa.get_duration(y=y)

        if (round(dur) < duration):
            input_length = sr * duration
            offset = len(y) - round(input_length)
            y = librosa.util.fix_length(y, round(input_length))

        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        if ps.shape != (128, 128):
            return np.zeros((1, 128, 128))

        return ps.reshape((1, 128, 128))

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''

    def input_y(self, Class):
        y_train_integer_encoded = self.label_to_int[Class]
        y_train = np.array(to_categorical(y_train_integer_encoded, len(self.list_labels)))

        return y_train

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def output_x(self, ID):
        print("output_x")
        return ID

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, data):
        print(data)

        return self.int_to_label[np.argmax(data)]
