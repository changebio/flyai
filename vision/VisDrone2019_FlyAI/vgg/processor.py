# -*- coding: utf-8 -*
from __future__ import print_function
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import sys
from time import time
import json
import os
import platform
import random
import requests
import pandas as pd
import numpy
import numpy as np

import cv2
from PIL import Image
from flyai.processor.base import Base
from flyai.processor.download import check_download
from flyai.utils.yaml_helper import Yaml
from flyai.utils import read_data

from path import DATA_PATH


image_resize = [800, 800]
output_size = [32, 32]
width_len = image_resize[1]//output_size[1]
height_len = image_resize[0]//output_size[0]
class_num = 11

label_list = list(range(10))
def make_weights_for_balanced_classes(labels, nclasses): 
    labels = [label_list.index(i) for i in labels]
    count = [0] * nclasses                                                      
    for item in labels:                                                         
        count[item] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/numpy.sqrt(count[i])                                 
    weight = [0] * len(labels)                                              
    for idx, val in enumerate(labels):                                          
        weight[idx] = weight_per_class[val]                                  
    return weight 



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


class Processor1(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def input_x(self, image_path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        path = check_download(image_path, DATA_PATH)
        path = path.replace('\\','/')
        image = Image.open(path)
        image = image.convert('RGB')
        image = image.resize((336, 336))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        pred_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        image = pred_transforms(image)
        #print('image process')
        return image

    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, landmark):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        landmark = landmark.strip('[]').split(',')
        landmark = [float(e) for e in landmark]
        #print(landmark,type(landmark))
        return landmark

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        return data.reshape([-1,18])
    
    #def output_y(self, datas):
    #    '''
    #    验证时使用，把模型输出的y转为对应的结果
    #    '''
    #    labels=[]
    #    for data in datas:
    #        labels.append(label_list[numpy.argmax(data)])
    #    return labels



class ImageLabel(Dataset):
    """Two colnum (image_path,label)

    Args:
        root (string): Root directory of dataset where directory
            ``caltech256`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root,df,transform=None, target_transform=None):
        self.root = root
        self.df = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image_path,annotations=self.df.iloc[index]
        path = check_download(image_path, DATA_PATH)
        images = cv2.imread(path)
        height, width, _ = images.shape
        images = cv2.resize(images, (image_resize[0], image_resize[1]), interpolation=cv2.INTER_CUBIC)
        
        annotations_path = check_download(annotations, DATA_PATH)
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
      
        if self.transform is not None:
            images = self.transform(images)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return images, np.array(label)

def Csv(config,line=""):
    if line is "":
        line = True
    else:
        line = False
    train_path = check_download(config['train_url'], DATA_PATH, is_print=line)
    data = read_data.read(train_path)
    val_path = check_download(config['test_url'], DATA_PATH, is_print=line)
    val = read_data.read(val_path)
    return data,val

def load_csv(custom_source=None):
    yaml = Yaml()
    try:
        f = open(os.path.join(sys.path[0], 'train.json'))
        line = f.readline().strip()
    except IOError:
        line = ""

    postdata = {'id': yaml.get_data_id(),
                'env': line,
                'time': time(),
                'sign': random.random(),
                'goos': platform.platform()}

    try:
        servers = yaml.get_servers()
        r = requests.post(servers[0]['url'] + "/dataset", data=postdata)
        source = json.loads(r.text)
    except:
        source = None

    if source is None:
        trn,val = Csv({'train_url': os.path.join(DATA_PATH, "dev.csv"),'test_url': os.path.join(DATA_PATH, "dev.csv")}, line)
    elif 'yaml' in source:
        source = source['yaml']
        if custom_source is None:
            trn,val = Csv(source['config'], line)
        else:
            source = custom_source
    else:
        if not os.path.exists(os.path.join(DATA_PATH, "train.csv")) and not os.path.exists(
                os.path.join(DATA_PATH, "test.csv")):
            raise Exception("invalid data id!")
        else:
            trn,val = Csv({'train_url': os.path.join(DATA_PATH, "train.csv"),'test_url': os.path.join(DATA_PATH, "test.csv")}, line)
    print(source)
    return trn,val


def load_data(combine=True,summary=True):
    trn,val = load_csv()
    if combine:
        trn = pd.concat([trn,val])
    if summary:
        data_summary = trn.describe()
        for k in range(data_summary.shape[1]):
            print(list(data_summary.iloc[:,k]))
        for i in range(1,trn.shape[1]):
            print(trn.iloc[:,i].value_counts()[:10])
    return trn, val