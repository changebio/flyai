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

from PIL import Image
from flyai.processor.base import Base
from flyai.processor.download import check_download
from flyai.utils.yaml_helper import Yaml
from flyai.utils import read_data

from path import DATA_PATH

class Processor(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def input_x(self, image_path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        path = check_download(image_path, DATA_PATH)
        image = Image.open(path)
        image = image.convert('RGB')
        #image = image.resize((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        pred_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        image = pred_transforms(image)
        #print('image process')
        return image

    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y,
                rightmouth_x, rightmouth_y):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        #input_y = numpy.array([lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y,
        #                 rightmouth_x, rightmouth_y])/[178, 218,178, 218,178, 218,178, 218,178, 218]
        input_y = (numpy.array([lefteye_x, lefteye_y, righteye_x, righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y,
                         rightmouth_x, rightmouth_y])-numpy.array([56,99,90,95,57,100,57,116,82,114]))/numpy.array([32, 30, 34, 27, 64, 56, 33, 58, 38, 59])
        return input_y

    def output_y(self, datas):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        labels=[]
        for data in datas:
            data[data>1]=1
            data[data<0]=0
            data = data*numpy.array([32, 30, 34, 27, 64, 56, 33, 58, 38, 59])+numpy.array([56,99,90,95,57,100,57,116,82,114])
            labels.append([numpy.around(data,decimals=0).astype('int')])
            
        return labels



class DatasetFlyAI(Dataset):
    def __init__(self, root, df, transform=None):
        self.root=root
        self.df=df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        one = self.df.iloc[index]
        image_path = one['image_path']
        label = (numpy.array(list(one[1:]))-numpy.array([56,99,90,95,57,100,57,116,82,114]))/numpy.array([32, 30, 34, 27, 64, 56, 33, 58, 38, 59])
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
