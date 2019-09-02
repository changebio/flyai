# -*- coding: utf-8 -*
from sklearn import preprocessing
import numpy
from flyai.processor.base import Base
from flyai.processor.download import check_download
from path import DATA_PATH
import json
import math

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
        path = check_download(image_path, DATA_PATH)
        with open(path) as f:
            data = json.load(f)
            frame_num = int(data['frame_num'])
            x_data = []
            for i in range(frame_num):
                mid = numpy.zeros([64,64])
                frame_i = data['frame_' + str(i)]
                for j in range(25):
                    x_r = frame_i['joint_' + str(j) + '_x:']
                    y_r = frame_i['joint_' + str(j) + '_y:']
                    if math.isnan(x_r) or math.isnan(y_r):
                        pass
                    else:
                        x = int(x_r/1920*64)
                        y = int(y_r/1080*64)
                        if x>63:
                            x = 63
                        if y>63:
                            y = 63
                        mid[x,y] = 1
                x_data.append(mid)
            x_data = numpy.array(x_data)
            x_data = x_data.reshape((50, 64, 64, 1))
            return x_data

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''

    def input_y(self, labels):
        one_hot_label = numpy.zeros([120])  ##生成全0矩阵
        one_hot_label[labels] = 1  ##相应标签位置置
        return one_hot_label

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def output_x(self, image_path):
        path = check_download(image_path, DATA_PATH)
        with open(path) as f:
            data = json.load(f)
            frame_num = int(data['frame_num'])
            x_data = []
            for i in range(frame_num):
                mid = numpy.zeros([64,64])
                frame_i = data['frame_' + str(i)]
                for j in range(25):
                    x_r = frame_i['joint_' + str(j) + '_x:']
                    y_r = frame_i['joint_' + str(j) + '_y:']
                    if math.isnan(x_r) or math.isnan(y_r):
                        pass
                    else:
                        x = int(x_r/1920*64)
                        y = int(y_r/1080*64)
                        if x>63:
                            x = 63
                        if y>63:
                            y = 63
                        mid[x,y] = 1
                x_data.append(mid)
            x_data = numpy.array(x_data)
            x_data = x_data.reshape((50, 64, 64, 1))
            return x_data

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, data):
        return numpy.argmax(data)
