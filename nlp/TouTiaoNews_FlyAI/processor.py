# -*- coding: utf-8 -*
from __future__ import print_function
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

def jiebatoken(text,w2d,words_len=30,char_len=30):
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
    return np.stack(vec_terms+char_terms)

class Processor(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def __init__(self):
        super(Processor, self).__init__()
        with open(DATA_PATH + '/embedding.json', 'r') as f:
            js = f.read()
            self.word_embed = json.loads(js)
        self.id2lael = ['news_culture', 'news_entertainment', 'news_sports', 'news_finance', 'news_house', 'news_car',
                        'news_edu', 'news_tech', 'news_military', 'news_travel', 'news_world', 'news_agriculture',
                        'news_game', 'stock', 'news_story']
        

    def input_x(self, news,keywords):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        #print(news,keywords)
        x_train = jiebatoken(news,self.word_embed)
        x2 = jiebatoken(keywords,self.word_embed)
        return x_train,x2

    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, category):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        labels = self.id2lael.index(category.strip('\n'))
        return labels

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        
        return self.id2lael[np.argmax(data)]

class NLPFlyAI(Dataset):
    def __init__(self, root, df, transform=None):
        self.root=root
        self.df=df
        self.transform = transform
        with open(DATA_PATH + '/embedding.json', 'r') as f:
            js = f.read()
            self.word_embed = json.loads(js)
        self.id2lael = ['news_culture', 'news_entertainment', 'news_sports', 'news_finance', 'news_house', 'news_car',
                        'news_edu', 'news_tech', 'news_military', 'news_travel', 'news_world', 'news_agriculture',
                        'news_game', 'stock', 'news_story']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        _,category,news,keywords = self.df.iloc[index]    
        x1 = jiebatoken(news,self.word_embed)
        x2 = jiebatoken(keywords,self.word_embed)
        label = self.id2lael.index(category.strip('\n'))
        return x1, x2, label
        

