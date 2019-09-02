# -*- coding: utf-8 -*
from __future__ import print_function
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import sys
from time import time
import os
import platform
import random
import requests
import pandas as pd
import numpy

import jieba
import codecs
import json
import numpy as np
import os
import re
from flyai.processor.base import Base
from flyai.processor.download import check_download
from flyai.utils.yaml_helper import Yaml
from flyai.utils import read_data

from bert import tokenization
from bert.run_classifier import convert_single_example_simple

from path import DATA_PATH  # 导入输入数据的地址

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

#sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
def words2vec(words,vocab,embedding_len,max_sts_len):
    words = str(words)
    words = jieba.cut(words, cut_all=False)
    
    vecs = []
    for word in words:
        embedding_vector = vocab.get(word)
        if embedding_vector is not None and len(embedding_vector) == embedding_len:
                vecs.append(embedding_vector)
    if len(vecs) >= max_sts_len:
        vecs = vecs[:max_sts_len]
    else:
        for i in range(len(vecs), max_sts_len):
            vecs.append([0 for j in range(embedding_len)])
    vecs = np.stack(vecs)
    return vecs

class Processor(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def __init__(self):
        super(Processor, self).__init__()
        self.token = None

    def input_x(self, texta, textb):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        if self.token is None:
            bert_vocab_file = os.path.join(DATA_PATH, "model", "uncased_L-12_H-768_A-12", 'vocab.txt')
            self.token = tokenization.CharTokenizer(vocab_file=bert_vocab_file)
        word_ids, word_mask, word_segment_ids = convert_single_example_simple(max_seq_length=32, tokenizer=self.token,
                                                                              text_a=texta, text_b=textb)
        return word_ids, word_mask, word_segment_ids


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
        labels = np.array(data)
        labels = labels.astype(np.float32)
        out_y = labels
        return out_y
class BertFlyAI(Dataset):
    def __init__(self, root,df,transform=None):
        self.root = root
        self.df = df
        self.token = None
      
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        _,texta,textb,label = self.df.iloc[index]  
        if self.token is None:
            bert_vocab_file = os.path.join(DATA_PATH, "model", "uncased_L-12_H-768_A-12", 'vocab.txt')
            self.token = tokenization.CharTokenizer(vocab_file=bert_vocab_file)
        word_ids, word_mask, word_segment_ids = convert_single_example_simple(max_seq_length=32, tokenizer=self.token,
                                                                              text_a=texta, text_b=textb)
        return word_ids, word_mask, word_segment_ids,label
    
class DQuestionFlyAI(Dataset):
    def __init__(self, root, df, transform=None):
        self.root=root
        self.df=df
        self.transform = transform
        embedding_path = os.path.join(DATA_PATH, 'embedding.json')
        with open(embedding_path, encoding='utf-8') as f:
            ch_vecs = json.loads(f.read())
        new_ch_vecs = dict()
        for k,v in ch_vecs.items():
            new_ch_vecs[k] = np.array([float(i) for i in v])
        self.vocab = new_ch_vecs
        self.max_sts_len = 20
        self.embedding_len = 200
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        _,question1,question2,label = self.df.iloc[index]    
        word_list1 = words2vec(question1,self.vocab,self.embedding_len,self.max_sts_len)
        word_list2 = words2vec(question2,self.vocab,self.embedding_len,self.max_sts_len)
        return word_list1, word_list2, label
        
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

