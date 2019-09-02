# -*- coding: utf-8 -*
from __future__ import print_function
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import json
import os
import numpy as np
from PIL import Image
from flyai.processor.base import Base
from flyai.processor.download import check_download

from path import DATA_PATH

NUM_KEYPOINT = 25
MAX_LEN = 1280  # 1280 /16
MAX_DEPTH = 32

class Processor(Base):
    
    def __init__(self):
        with open(os.path.join(DATA_PATH,'label.json')) as fin:
            self.char_dict_res_ =json.loads(fin.read())
        self.char_dict = dict()
        self.char_dict_res = dict()
        for i,word in self.char_dict_res_.items():
            self.char_dict[word] = i
            self.char_dict_res[int(i)]=word
            
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def input_x(self, path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        path = check_download(path, DATA_PATH)
        x = np.zeros((MAX_DEPTH, NUM_KEYPOINT, 2), dtype='float32')
        with open(path) as fin:
            di = 0
            for line in fin:
                terms = line.strip().split()
                terms = [int(term) for term in terms]
                for i in range(len(terms) // 2):
                    h_origin = int(terms[i * 2])
                    w_origin = int(terms[i * 2 + 1])
                    if h_origin != 0 or w_origin != 0:
                        h = h_origin
                        w = w_origin
                        x[di, i, 0] = (h * 1.0 - MAX_LEN // 2) / MAX_LEN
                        x[di, i, 1] = (w * 1.0 - MAX_LEN // 2) / MAX_LEN
                di += 1
                if di >= MAX_DEPTH:
                    break
        xdiff = np.zeros((MAX_DEPTH, NUM_KEYPOINT, 2), dtype='float32')
        xdiff[:-1, :, :] = xdiff[1:, :, :] - xdiff[:-1, :, :]
        return np.concatenate((x, xdiff),axis=2).transpose(2,0,1)
    
    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, label):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        label = int(self.char_dict.get(label,"0"))
        return label

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        return self.char_dict_res.get(np.argmax(data),"<unk>")

class SLR1FlyAI(Dataset):
    def __init__(self, root, df, transform=None):
        self.root=root
        self.df=df
        self.transform = transform
        with open(os.path.join(DATA_PATH,'label.json')) as fin:
            self.char_dict_res_ =json.loads(fin.read())
        self.char_dict = dict()
        self.char_dict_res = dict()
        for i,word in self.char_dict_res_.items():
            self.char_dict[word] = i
            self.char_dict_res[int(i)]=word
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        one = self.df.iloc[index]
        path = one[0]
        label = one[1]
        path = check_download(path, DATA_PATH)
        x = np.zeros((MAX_DEPTH, NUM_KEYPOINT, 2), dtype='float32')
        with open(path) as fin:
            di = 0
            for line in fin:
                terms = line.strip().split()
                terms = [int(term) for term in terms]
                for i in range(len(terms) // 2):
                    h_origin = int(terms[i * 2])
                    w_origin = int(terms[i * 2 + 1])
                    if h_origin != 0 or w_origin != 0:
                        h = h_origin
                        w = w_origin
                        x[di, i, 0] = (h * 1.0 - MAX_LEN // 2) / MAX_LEN
                        x[di, i, 1] = (w * 1.0 - MAX_LEN // 2) / MAX_LEN
                di += 1
                if di >= MAX_DEPTH:
                    break
        xdiff = np.zeros((MAX_DEPTH, NUM_KEYPOINT, 2), dtype='float32')
        xdiff[:-1, :, :] = xdiff[1:, :, :] - xdiff[:-1, :, :]
        label = int(self.char_dict.get(label,"0"))
        
        if self.transform is not None:
            image = self.transform(image)
            
        return np.concatenate((x, xdiff),axis=2).transpose(2,0,1),label

        
        

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
        label = one['label']
        #image_path,label,_ = self.df.iloc[index]
        path = check_download(image_path, self.root)
        image = Image.open(path)
        image = image.convert('RGB')

        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

