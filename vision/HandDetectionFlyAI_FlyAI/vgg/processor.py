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

from PIL import Image
from flyai.processor.base import Base
from flyai.processor.download import check_download
from flyai.utils.yaml_helper import Yaml
from flyai.utils import read_data

from path import DATA_PATH

class Processor(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def __init__(self):
        self.img_size = [224, 224]
        
    def input_x(self, image_path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        path = check_download(image_path, DATA_PATH)
        image = Image.open(path)
        image = image.convert('RGB')
        image = image.resize((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        pred_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        image = pred_transforms(image)
        #print('image process')
        return image

    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, p1,p2, p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        height_scale = p1 / self.img_size[0]
        width_scale = p2 / self.img_size[1]
        # 以图片高度的方向为y方向
        all_bb_y = np.array([p3, p5, p7, p9, p11, p13, p15, p17, p19, p21, p23, p25])
        all_bb_x = np.array([p4, p6, p8, p10, p12, p14, p16, p18, p20, p22, p24, p26])
        # resize并归一化
        new_bb_x = (all_bb_x / width_scale) / self.img_size[1]
        new_bb_y = (all_bb_y / height_scale) / self.img_size[0]
        label = numpy.concatenate([new_bb_x,new_bb_y])
        return label
       

    def output_y(self, datas):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        labels=[]
        for data in datas:
            data[data>1]=1
            data[data<0]=0
            data = data*224
            labels.append([numpy.around(data,decimals=0).astype('int')])
            
        return labels



class DatasetFlyAI(Dataset):
    def __init__(self, root, df, transform=None):
        self.root=root
        self.df=df
        self.transform = transform
        self.img_size = [224, 224]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        one = self.df.iloc[index]
        image_path = one['image_path']
        p1,p2, p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26 = one[1:]
        height_scale = p1 / self.img_size[0]
        width_scale = p2 / self.img_size[1]
        # 以图片高度的方向为y方向
        all_bb_y = np.array([p3, p5, p7, p9, p11, p13, p15, p17, p19, p21, p23, p25])
        all_bb_x = np.array([p4, p6, p8, p10, p12, p14, p16, p18, p20, p22, p24, p26])
        # resize并归一化
        new_bb_x = (all_bb_x / width_scale) / self.img_size[1]
        new_bb_y = (all_bb_y / height_scale) / self.img_size[0]
        label = numpy.concatenate([new_bb_x,new_bb_y])
        #image_path,label,_ = self.df.iloc[index]
        path = check_download(image_path, self.root)
        image = Image.open(path)
        image = image.convert('RGB')

        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


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
