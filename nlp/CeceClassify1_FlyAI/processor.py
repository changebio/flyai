# -*- coding: utf-8 -*
from __future__ import print_function
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import sys

import jieba
import create_dict
import numpy as np
import os
import re
from flyai.processor.base import Base

from path import DATA_PATH  # 导入输入数据的地址
MAX_LEN = 20
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
def words2vec(words,vocab,embedding_len,max_sts_len):
    words = str(words)
    words = re.sub("[\s+\.\!\/_,$%^*()+-?\"\']+|[+——！，。；？、~@#￥%……&*（）]+", " ", words)
    words = words.strip().split(' ')
    
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

def tokenize(text,w2d,MAX_LEN=MAX_LEN):
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

class Processor(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def __init__(self):
        super(Processor, self).__init__()
        self.word_dict, _ = create_dict.load_dict()
        

    def input_x(self, text):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        index_list,char_index_list = tokenize(text,self.word_dict)
        return index_list, char_index_list

    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, labels):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        return labels

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        labels = np.array(data)
        labels = labels.astype(np.float32)
        out_y = labels
        return out_y

class NLPFlyAI(Dataset):
    def __init__(self, root, df, transform=None):
        self.root=root
        self.df=df
        self.transform = transform
        self.word_dict, _ = create_dict.load_dict()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text,label = self.df.iloc[index]    
        index_list,char_index_list = tokenize(text,self.word_dict)
        return index_list, char_index_list, label
        

