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
list_labels = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
                            'Small-flowered Cranesbill', 'Fat Hen', 'Loose Silky-bent', 'Maize',
                            'Scentless Mayweed', 'Shepherds Purse', 'Sugar beet']
# Map integer value to text labels
label_to_int = {k: v for v, k in enumerate(list_labels)}
def make_weights_for_balanced_classes(labels, nclasses):                        
    count = [0] * nclasses                                                      
    labels = [label_to_int[i] for i in labels]
    for item in labels:                                                         
        count[item] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):
        if count[i]!=0:
            weight_per_class[i] = N/np.sqrt(count[i])                                 
    weight = [0] * len(labels)                                              
    for idx, val in enumerate(labels):                                          
        weight[idx] = weight_per_class[val]                                  
    return weight 

class Processor(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def __init__(self):
        self.list_labels = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
                            'Small-flowered Cranesbill', 'Fat Hen', 'Loose Silky-bent', 'Maize',
                            'Scentless Mayweed', 'Shepherds Purse', 'Sugar beet']
        # Map integer value to text labels
        self.label_to_int = {k: v for v, k in enumerate(self.list_labels)}

        # map integer to text labels
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}
    def input_x(self, path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        path = check_download(path, DATA_PATH)
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
    def input_y(self, seedling):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        return self.label_to_int[seedling]

    def output_x(self, path):
        return path
    
    def output_y(self, datas):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        labels=[]
        for data in datas:
            labels.append(self.int_to_label[numpy.argmax(data)])
        return labels



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
        self.list_labels = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
                            'Small-flowered Cranesbill', 'Fat Hen', 'Loose Silky-bent', 'Maize',
                            'Scentless Mayweed', 'Shepherds Purse', 'Sugar beet']
        # Map integer value to text labels
        self.label_to_int = {k: v for v, k in enumerate(self.list_labels)}

        # map integer to text labels
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path,label=self.df.iloc[index]
        path = check_download(img_path, self.root)
        image = Image.open(path)
        image = image.convert('RGB')
      
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)
        print(image.shape)
        return image, self.label_to_int[label]

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
