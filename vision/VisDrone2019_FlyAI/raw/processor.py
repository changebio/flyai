# -*- coding: utf-8 -*
from flyai.processor.base import Base
from flyai.processor.download import check_download
from config import image_resize, class_num, output_size, width_len, height_len
from path import DATA_PATH
import cv2
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

    def input_x(self, image_path):
        # 固定网络输入大小
        path = check_download(image_path, DATA_PATH)
        images = cv2.imread(path)
        images = cv2.resize(images, (image_resize[0], image_resize[1]), interpolation=cv2.INTER_CUBIC)
        images = images / 255
        return images

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''

    def input_y(self, image_path, annotations):
        image_path = check_download(image_path, DATA_PATH)
        annotations_path = check_download(annotations, DATA_PATH)
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        height_scale = height / image_resize[0]
        width_scale = width / image_resize[1]
        # 构建标签
        label = np.zeros((output_size[0], output_size[1], class_num + 4))
        label[:, :, 0] = 1
        label_mask = np.zeros((output_size[0], output_size[1]))
        with open(annotations_path) as f:
            for line in f.readlines():
                line_data = np.array(line.strip('\n').split(',')).astype(int)
                # <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                new_left = int(line_data[0] / width_scale)
                new_top = int(line_data[1] / height_scale)
                new_width = line_data[2] / width_scale
                new_height = line_data[3] / height_scale
                new_width_center = new_left + int(new_width / 2)
                new_height_center = new_top + int(new_height / 2)
                index_x = new_width_center // width_len
                index_y = new_height_center // height_len
                if (label_mask[index_y][index_x] == 0 and line_data[4] == 1 and line_data[5] not in [0, 11]):
                    label_mask[index_y][index_x] = 1
                    label[index_y][index_x][0] = 0
                    label[index_y][index_x][line_data[5]] = 1
                    label[index_y][index_x][-4] = (new_width_center % width_len) / width_len
                    label[index_y][index_x][-3] = (new_height_center % height_len) / height_len
                    label[index_y][index_x][-2] = np.log(new_width)
                    label[index_y][index_x][-1] = np.log(new_height)
        return label

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, annotations):
        return
