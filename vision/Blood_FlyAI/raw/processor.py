# -*- coding: utf-8 -*
import cv2
import numpy
from flyai.processor.base import Base
from flyai.processor.download import check_download

from path import DATA_PATH


class Processor(Base):

    def input_x(self, image_path):
        path = check_download(image_path, DATA_PATH)
        image = cv2.imread(path)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        x_data = numpy.array(image)
        x_data = x_data.astype(numpy.float32)
        x_data = numpy.multiply(x_data, 1.0 / 255.0)
        return x_data

    def input_y(self, labels):
        one_hot_label = numpy.zeros([4])  ##生成全0矩阵
        one_hot_label[labels] = 1  ##相应标签位置置
        return one_hot_label

    def output_y(self, data):
        return int(numpy.argmax(data))
