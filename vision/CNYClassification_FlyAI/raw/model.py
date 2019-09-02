# -*- coding: utf-8 -*
import os
import tensorflow as tf
from flyai.model.base import Base
from path import MODEL_PATH
import numpy as np


TENSORFLOW_MODEL_DIR = "model.ckpt"

class Model(Base):
    def __init__(self, data):
        self.data = data
        self.model_path = MODEL_PATH
        self.label_list = [2.0, 0.2, 10.0, 50.0, 0.5, 100.0, 0.1, 5.0, 1.0]
        self.img_shape = [224, 224, 3]
        self.is_load = False
        self.latest_ckpt = tf.train.latest_checkpoint(self.model_path)
        # 构建模型
        self.my_model = tf.keras.applications.ResNet50(input_shape=(self.img_shape[0], self.img_shape[1], 3),
                                                       weights=None,
                                                       include_top=True, classes=len(self.label_list))
        #self.my_model.summary()
        if self.latest_ckpt:
            print('load  my model from __init__  !!!!!!!!!!!!!')
            self.my_model.load_weights(self.latest_ckpt)
            self.is_load = True

    def predict(self, **data):
        '''
        使用模型
        :param path: 模型所在的路径
        :param name: 模型的名字
        :param data: 模型的输入参数
        :return:
        '''
        if not self.is_load:
            print('load  my model from predict  !!!!!!!!!!!!!')
            self.my_model = tf.keras.applications.ResNet50(input_shape=(self.img_shape[0], self.img_shape[1], 3),
                                                           weights=None,
                                                           include_top=True, classes=len(self.label_list))

            self.my_model.load_weights(self.latest_ckpt)
            self.is_load = True
        x_data = self.data.predict_data(**data)
        predict = self.my_model.predict(x_data)
        index = np.argmax(predict[0])
        return self.label_list[index]

    def predict_all(self, datas):
        if not self.is_load:
            print('load  my model from predict_all !!!!!!!!!!!!!')
            self.my_model = tf.keras.applications.ResNet50(input_shape=(self.img_shape[0], self.img_shape[1], 3),
                                                           weights=None,
                                                           include_top=True, classes=len(self.label_list))

            self.my_model.load_weights(self.latest_ckpt)
            self.is_load = True
        labels = []
        for data in datas:
            x_data = self.data.predict_data(**data)
            predict = self.my_model.predict(x_data)
            index = np.argmax(predict[0])
            labels.append(self.label_list[index])
        return labels

    def save_model(self, mymodel, path, name=TENSORFLOW_MODEL_DIR, overwrite=True):
        '''
        保存模型
        :param session: 训练模型的sessopm
        :param path: 要保存模型的路径
        :param name: 要保存模型的名字
        :param overwrite: 是否覆盖当前模型
        :return:
        '''
        super().save_model(mymodel, path, name, overwrite)
        mymodel.save_weights(os.path.join(MODEL_PATH, name))




