# -*- coding: utf-8 -*
import numpy as np
from flyai.processor.base import Base
from flyai.processor.download import check_download
from skimage import io, transform

from path import DATA_PATH


class Processor(Base):

    def input_x(self, image_path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        path = check_download(image_path, DATA_PATH)
        images = io.imread(path)
        image = transform.resize(images, (178, 218))  # 改变图片的大小:178*218
        image = np.array(image)
        image = image.astype(np.float32)  # 传入卷积神经网络中时的形状
        input_x = image

        return input_x

    def input_y(self, lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y,
                rightmouth_x, rightmouth_y):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        temp = np.array([lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y,
                         rightmouth_x, rightmouth_y])
        input_y = temp.astype(np.float32)
        return input_y

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        return data
