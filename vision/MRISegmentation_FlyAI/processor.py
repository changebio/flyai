# -*- coding: utf-8 -*
import numpy as np
from flyai.processor.base import Base
import cv2
from path import DATA_PATH
import os
'''
把样例项目中的processor.py件复制过来替换即可
'''


class Processor(Base):
    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def input_x(self, image_path):
        img_path = os.path.join(DATA_PATH, image_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        return img

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''

    def input_y(self, label_path):
        img_path = os.path.join(DATA_PATH, label_path)
        img_mask = cv2.imread(img_path)
        label_1 = img_mask[:, :, 0] / 255.0
        label_0 = 1 - label_1
        label = np.zeros((img_mask.shape[0], img_mask.shape[1], 2), dtype=np.int)
        label[:, :, 0] = label_0
        label[:, :, 1] = label_1
        return label

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def output_x(self, image_path):
        img_path = os.path.join(DATA_PATH, image_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        return img

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, pred_onehot):
        pred_onehot = pred_onehot[0]
        h, w, _ = pred_onehot.shape
        pred_label = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                index = np.argmax(pred_onehot[i,j])
                if(index == 1):
                    pred_label[i][j] = 1
        return pred_label
