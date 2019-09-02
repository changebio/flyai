# -*- coding: utf-8 -*
from __future__ import print_function
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import sys

import codecs
import json
import numpy as np
import os
import re
from flyai.processor.base import Base

from path import DATA_PATH  # 导入输入数据的地址

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

class Processor(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def __init__(self):
        super(Processor, self).__init__()
        embedding_path = os.path.join(DATA_PATH, 'embedding.json')
        with open(embedding_path, encoding='utf-8') as f:
            self.vocab = json.loads(f.read())
        self.max_sts_len = 10
        self.embedding_len = 100

    def input_x(self, question1, question2):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        word_list1 = words2vec(question1,self.vocab,self.embedding_len,self.max_sts_len)
        word_list2 = words2vec(question2,self.vocab,self.embedding_len,self.max_sts_len)
        return word_list1, word_list2

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

class DQuestionFlyAI(Dataset):
    def __init__(self, root, df, transform=None):
        self.root=root
        self.df=df
        self.transform = transform
        embedding_path = os.path.join(DATA_PATH, 'embedding.json')
        with open(embedding_path, encoding='utf-8') as f:
            self.vocab = json.loads(f.read())
        self.max_sts_len = 10
        self.embedding_len = 100
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        if self.transform is not None:
            image = self.transform(image)
        question1,question2,label = self.df.iloc[index]    
        word_list1 = words2vec(question1,self.vocab,self.embedding_len,self.max_sts_len)
        word_list2 = words2vec(question2,self.vocab,self.embedding_len,self.max_sts_len)
        return word_list1, word_list2, label
        

