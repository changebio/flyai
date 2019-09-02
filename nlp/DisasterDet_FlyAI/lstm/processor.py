# -*- coding: utf-8 -*
from __future__ import print_function
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import sys
import json

import jieba
import create_dict
import numpy as np
import os
import re
from flyai.processor.base import Base
from math import isnan

from time import time
import platform
import random
import requests
import pandas as pd
import numpy
import codecs
from flyai.processor.download import check_download
from flyai.utils.yaml_helper import Yaml
from flyai.utils import read_data

from path import DATA_PATH  # 导入输入数据的地址
#from pytorch_pretrained_bert import BertTokenizer
# Load pre-trained model tokenizer (vocabulary)
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def make_weights_for_balanced_classes(labels, nclasses):                        
    count = [0] * nclasses                                                      
    for item in labels:                                                         
        count[item] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(np.sqrt(count[i]))
    #weight_per_class[0] = 0.3
    #weight_per_class[1] = 0.7
    weight = [0] * len(labels)                                              
    for idx, val in enumerate(labels):                                          
        weight[idx] = weight_per_class[val]                                  
    return weight 


MAX_LEN = 20
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
def words2vec(words,vocab,embedding_len,max_sts_len):
    words = str(words)
    words = re.sub("[\s+\.\!\/_,$%^*()+-?\"\']+|[+——！，。；？、~@#￥%……&*（）]+", " ", words)
    words = words.strip().split(' ')
    
    vecs = []
    for word in words:
        embedding_vector = vocab.get(word)
        if embedding_vector is not None:
                vecs.append(embedding_vector)
    if len(vecs) >= max_sts_len:
        vecs = vecs[:max_sts_len]
    elif embedding_len>1:
        for i in range(len(vecs), max_sts_len):
            vecs.append([0.0 for j in range(embedding_len)])
    else:
        for i in range(len(vecs), max_sts_len):
            vecs.append(0)
    vecs = np.stack(vecs)
    if embedding_len ==1:
        vecs = vecs.reshape(max_sts_len,1)
    return vecs

def tokenize(text,w2d,max_len):
    text = str(text)
    terms = jieba.cut(text, cut_all=False)
    truncate_terms = []
    for term in terms:
        truncate_terms.append(term)
        if len(truncate_terms) >= MAX_LEN:
            break
    index_list = [w2d[term] if term in w2d
                  else create_dict._UNK_ for term in truncate_terms]
    if len(index_list) < MAX_LEN:
        index_list = index_list + [create_dict._PAD_] * (MAX_LEN - len(index_list))

    char_index_list = [w2d[c] if c in w2d
                       else create_dict._UNK_ for c in text]
    char_index_list = char_index_list[:MAX_LEN]
    if len(char_index_list) < MAX_LEN:
        char_index_list = char_index_list + [create_dict._PAD_] * (MAX_LEN - len(char_index_list))
    return np.array(index_list)[:,None], np.array(char_index_list)[:,None]

def jiebatoken(text,w2d,words_len=30,char_len=30,stack=True):
    text = str(text)
    fill_embed = [0.0 for i in range(200)]
    terms = jieba.cut(text, cut_all=False)
    vec_terms = []
    for term in terms:
        if term in w2d.keys() and len(vec_terms) < words_len:
            vec_terms.append([float(i) for i in w2d[term]])
    for j in range(words_len - len(vec_terms)):
        vec_terms.append(fill_embed)
    vec_terms = vec_terms[:words_len]
    
    char_terms = []       
    for term in text:
        if term in w2d.keys() and len(vec_terms) < char_len:
            char_terms.append([float(i) for i in w2d[term]])
    for j in range(char_len - len(char_terms)):
        char_terms.append(fill_embed)
    char_terms = char_terms[:char_len]
    if stack:
        return np.stack(vec_terms+char_terms)
    else:
        return np.stack(vec_terms),np.stack(char_terms)

class Processor(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def __init__(self):
        super(Processor, self).__init__()
        try:
            embedding_path = os.path.join(DATA_PATH, 'embedding.json')
            with open(embedding_path) as f:
                ch_vecs = json.loads(f.read())

            new_ch_vecs = dict()
            for k,v in ch_vecs.items():
                new_ch_vecs[k] = np.array([float(i) for i in v])
            self.w2v = new_ch_vecs
        except:
            print("no embeding")
        self.w2d, _ = create_dict.load_dict()
        self.max_sts_len = 30
        self.embedding_len = 1
       
    def input_x(self, text):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        vec = words2vec(text,self.w2d,self.embedding_len,self.max_sts_len)
        #token = words2vec(sentence,self.w2d,1,self.max_sts_len)
        
        return vec

    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, label):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        return label

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        
        return np.argmax(data)

class NLPFlyAI(Dataset):
    def __init__(self, root, df, transform=None):
        self.root=root
        self.df=df
        try:
            embedding_path = os.path.join(DATA_PATH, 'embedding.json')
            with open(embedding_path) as f:
                ch_vecs = json.loads(f.read())

            new_ch_vecs = dict()
            for k,v in ch_vecs.items():
                new_ch_vecs[k] = np.array([float(i) for i in v])
            self.w2v = new_ch_vecs
        except:
            print("no embeding")
        self.w2d, _ = create_dict.load_dict()
        self.max_sts_len = 30
        self.embedding_len = 1
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        label,text = self.df.iloc[index]    
        vec = words2vec(text,self.w2d,self.embedding_len,self.max_sts_len)
        #token = words2vec(text,self.w2d,1,self.max_sts_len)
        #print(text,label,vec.shape)
        
        return vec,label
        
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

