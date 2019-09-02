import numpy as np
import cv2
from flyai.processor.base import Base
from flyai.processor.download import check_download
from path import DATA_PATH
import json


class Processor(Base):
    def __init__(self):
        # 训练的时候做数据增强用
        self.img_augment = dict()
        self.img_size = [224, 224]

    def input_x(self, image_path):
        path = check_download(image_path, DATA_PATH)
        images = cv2.imread(path)
        # 随机偏移
        rand_x = np.random.randint(0, 50, 1)[0]
        rand_y = np.random.randint(0, 50, 1)[0]
        M = np.float32([[1, 0, rand_x], [0, 1, rand_y]])
        images = cv2.warpAffine(images, M, (images.shape[1], images.shape[0]))
        self.img_augment[image_path] = [rand_x, rand_y]

        images = cv2.resize(images, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_CUBIC)
        images = images / 255
        return images

    def input_y(self, image_path, p1,p2, p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26):
        '''
        :param p1:  The height of original image
        :param p2:  The width of original image
        :param p3 - p6:  hand bounding box.
        :param p7 - p10: palm bounding box.
        :param p11 - p14: index bounding box.
        :param p15 - p18:  middle bounding box.
        :param p19 - p22:  ring bounding box.
        :param p23 - p26:  little bounding box.
        :return:  label
        '''
        height_scale = p1 / self.img_size[0]
        width_scale = p2 / self.img_size[1]
        # 以图片高度的方向为y方向
        all_bb_y = np.array([p3, p5, p7, p9, p11, p13, p15, p17, p19, p21, p23, p25])
        all_bb_x = np.array([p4, p6, p8, p10, p12, p14, p16, p18, p20, p22, p24, p26])
        # 标签也做同样的增强变换
        all_bb_x = all_bb_x + self.img_augment[image_path][0]
        all_bb_y = all_bb_y + self.img_augment[image_path][1]
        # resize并归一化
        new_bb_x = (all_bb_x / width_scale) / self.img_size[1]
        new_bb_y = (all_bb_y / height_scale) / self.img_size[0]
        label = []
        label.extend(new_bb_x)
        label.extend(new_bb_y)
        return label

    # 预测时，不做数据增强（这个函数不需要修改）
    def output_x(self, image_path):
        path = check_download(image_path, DATA_PATH)
        images = cv2.imread(path)
        images = cv2.resize(images, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_CUBIC)
        images = images / 255
        return images
