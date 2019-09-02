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
import re
import zipfile

from PIL import Image
from flyai.processor.base import Base
from flyai.processor.download import check_download
from flyai.utils.yaml_helper import Yaml
from flyai.utils import read_data

from path import DATA_PATH

import librosa
from joblib import Parallel, delayed
import multiprocessing

# Map integer value to text labels
label_to_int = {'COPD': 0,'Healthy': 1,'URTI': 2,'Bronchiectasis': 3,'Pneumonia': 4,'Bronchiolitis': 5,'LRTI': 6}
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
        # Map integer value to text labels
        self.label_to_int =  {'COPD': 0,'Healthy': 1,'URTI': 2,'Bronchiectasis': 3,'Pneumonia': 4,'Bronchiolitis': 5,'LRTI': 6}

        # map integer to text labels
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}
    def input_x(self, audio_and_txt_files_path,patient_id,age,sex,adult_bmi,child_weight,child_height):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        #audio_and_txt_files_path="audio_and_txt_files/182.zip"
        path_list = audio_and_txt_files_path.split('/')
        path = path_list[1].split('.')[0]
        files = zipfile.ZipFile(DATA_PATH + '/' + audio_and_txt_files_path, 'r')
        if not os.path.exists(DATA_PATH + '/' + path):
            for file in files.namelist():
                files.extract(file, DATA_PATH + '/' + path + '/')

            wav_path = os.listdir(DATA_PATH + '/' + path)
            wav_list = []
            wav = path + '/'
            sounds = []

            for wavs in wav_path:
                wav = path + '/'
                if re.match('.*.wav', wavs) != None:
                    duration = 2.97
                    sr = 22050
                    y, sr = librosa.load(DATA_PATH + '/' + path + '/' + wavs, duration=duration, sr=sr)
                    ps = librosa.feature.melspectrogram(y=y, sr=sr)
                    dur = librosa.get_duration(y=y)
                    break
        else:
            wav_path = os.listdir(DATA_PATH + '/' + path)
            wav_list = []
            wav = path + '/'
            sounds = []

            for wavs in wav_path:
                wav = path + '/'
                if re.match('.*.wav', wavs) != None:
                    duration = 2.97
                    sr = 22050
                    y, sr = librosa.load(DATA_PATH + '/' + path + '/' + wavs, duration=duration, sr=sr)
                    ps = librosa.feature.melspectrogram(y=y, sr=sr)
                    dur = librosa.get_duration(y=y)
                    break



        if (round(dur) < duration):
            input_length = sr * duration
            offset = len(y) - round(input_length)
            y = librosa.util.fix_length(y, round(input_length))
            ps = librosa.feature.melspectrogram(y=y, sr=sr)

        if ps.shape != (128, 128):
            return np.zeros((128, 128))

        return ps


    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, diagnosis):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        return self.label_to_int[diagnosis]

    
    
    def output_y(self, datas):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        labels=[]
        for data in datas:
            labels.append(self.int_to_label[numpy.argmax(data)])
        return labels

def read_wav(f,dr=10):
    sr = 22050
    y_len = sr*dr
    #print(f)
    try:
        y,_ = librosa.load(os.path.join(DATA_PATH, f))
        yl = len(y)
        if yl<y_len:
            y = np.tile(y,np.ceil(y_len/yl).astype(int))[:y_len]
        elif yl>y_len:
            y = y[:y_len]
    except:
        y = np.zeros(y_len,dtype=np.float32)
        print("error: loading wav")
    return y

def read_one_zip_wav(audio_and_txt_files_path):
    path_list = audio_and_txt_files_path.split('/')
    path = path_list[1].split('.')[0]
    files = zipfile.ZipFile(DATA_PATH + '/' + audio_and_txt_files_path, 'r')
    if not os.path.exists(DATA_PATH + '/' + path):
        for file in files.namelist():
            files.extract(file, DATA_PATH + '/' + path + '/')

    wav_path = os.listdir(DATA_PATH + '/' + path)
    for wavs in wav_path:
        wav = path + '/'
        if re.match('.*.wav', wavs) != None:
            return read_wav(DATA_PATH + '/' + path + '/' + wavs)
    

def read_df(df):
    wf = []
    labels = []
    for i in range(len(df)):
        #print(df.iloc[i])
        audio_and_txt_files_path,patient_id,age,sex,adult_bmi,child_weight,child_height,diagnosis = df.iloc[i]
        path_list = audio_and_txt_files_path.split('/')
        path = path_list[1].split('.')[0]
        files = zipfile.ZipFile(DATA_PATH + '/' + audio_and_txt_files_path, 'r')
        if not os.path.exists(DATA_PATH + '/' + path):
            for file in files.namelist():
                files.extract(file, DATA_PATH + '/' + path + '/')

        wav_path = os.listdir(DATA_PATH + '/' + path)
        
        for wavs in wav_path:
            if re.match('.*.wav', wavs) != None:
                wf.append(DATA_PATH + '/' + path + '/' + wavs)
                labels.append(diagnosis)
                
    return wf,labels

    
def predict_data(fs):
    num_cores = multiprocessing.cpu_count()
    print('num_cores',num_cores)
    wav = Parallel(n_jobs=num_cores)(delayed(read_one_zip_wav)(f) for f in fs)
    print('loading test done')
    ps = [librosa.feature.melspectrogram(y).transpose() for y in wav]
    return np.stack(ps)

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
        self.wf,self.labels = read_df(self.df)
        self.transform = transform
        self.target_transform = target_transform
        self.label_to_int = {'COPD': 0,'Healthy': 1,'URTI': 2,'Bronchiectasis': 3,'Pneumonia': 4,'Bronchiolitis': 5,'LRTI': 6}


        # map integer to text labels
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}
        num_cores = multiprocessing.cpu_count()
        print('num_cores',num_cores)
        self.wav = Parallel(n_jobs=num_cores)(delayed(read_wav)(f) for f in self.wf)
        print('done')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        label=self.labels[index]
        ps = librosa.feature.melspectrogram(self.wav[index])
        if self.transform is not None:
            ps = self.transform(ps)

        if self.target_transform is not None:
            label = self.target_transform(label)
        return ps.transpose(),self.label_to_int[label]
        

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
        try:
            for k in range(data_summary.shape[1]):
                print(list(data_summary.iloc[:,k]))
            for i in range(1,trn.shape[1]):
                print(trn.iloc[:,i].value_counts()[:10])
        except:
            print("somethings wrong with the data") 
    return trn, val
