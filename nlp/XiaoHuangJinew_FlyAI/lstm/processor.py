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

import jieba
import create_dict
import numpy as np
import re
from math import isnan

from PIL import Image
from flyai.processor.base import Base
from flyai.processor.download import check_download
from flyai.utils.yaml_helper import Yaml
from flyai.utils import read_data

from path import DATA_PATH


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
            vecs.append([0 for j in range(embedding_len)])
    else:
        for i in range(len(vecs), max_sts_len):
            vecs.append(0)
    vecs = np.stack(vecs)
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
    def __init__(self):
        with open(os.path.join(DATA_PATH, 'words.dict'), 'r') as f:
            js = f.read()
            self.word2id_dict = json.loads(js)
        self.id2word_dict = {}
        for k, v in self.word2id_dict.items():
            self.id2word_dict[v] = k
        self.length = len(self.word2id_dict) + 5
        # 空值填充0
        self.PAD_ID = 1
        # 输出序列起始标记
        self.GO_ID = 0
        # 结尾标记
        self.EOS_ID = 2
        self.UNK = 3
        self.input_seq_len = 10
        self.output_seq_len = 10

    def input_x(self, question):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        x_ids = self.get_id_list_from(question)
        cut = self.input_seq_len
        x_ids_tr = np.array(x_ids[:cut] if len(x_ids) > cut else x_ids+[self.PAD_ID]*(cut-len(x_ids)))
        return x_ids_tr

    def input_y(self, answer):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        y_ids = self.get_id_list_from(answer)
        cut = self.output_seq_len
        y_ids_tr = np.array(y_ids[:cut] if len(y_ids) > cut else y_ids+[self.PAD_ID]*(cut-len(y_ids)))
        return y_ids_tr

    def output_y(self, outputs_seq):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        sent = []
        for v in outputs_seq:
            if v != self.EOS_ID:
                sent.append(self.id2word(v))
            else:
                break
       
        return "".join(sent)

    def id2word(self, id):
        id = int(id)
        if id in self.id2word_dict:
            return self.id2word_dict[id]
        else:
            return ""

    def get_id_list_from(self, sentence):
        sentence_id_list = []
        seg_list = jieba.cut(sentence)
        for str in seg_list:
            sentence_id_list.append(self.word2id(str))
        return sentence_id_list

    def word2id(self, word):
        if not isinstance(word, str):
            print("Exception: error word not unicode")
            sys.exit(1)
        if word in self.word2id_dict:
            return self.word2id_dict[word]
        else:
            return 3  # UNK

    def get_samples(self, train_set, input_seq_len, output_seq_len):
        """构造样本数据

        :return:
            encoder_inputs: [array([0, 0], dtype=int32), array([0, 0], dtype=int32), array([5, 5], dtype=int32),
                            array([7, 7], dtype=int32), array([9, 9], dtype=int32)]
            decoder_inputs: [array([1, 1], dtype=int32), array([11, 11], dtype=int32), array([13, 13], dtype=int32),
                            array([15, 15], dtype=int32), array([2, 2], dtype=int32)]
        """

        raw_encoder_input = []
        raw_decoder_input = []

        for i in range(len(train_set[0])):
            raw_encoder_input.append([self.PAD_ID] * (input_seq_len - len(train_set[0][i])) + list(train_set[0][i]))
            raw_decoder_input.append(
                [self.GO_ID] + list(train_set[1][i]) + [self.PAD_ID] * (output_seq_len - len(train_set[1][i]) - 1))

        encoder_inputs = []
        decoder_inputs = []
        target_weights = []

        for length_idx in range(input_seq_len):
            encoder_inputs.append(
                np.array([encoder_input[length_idx] for encoder_input in raw_encoder_input], dtype=np.int32))
        for length_idx in range(output_seq_len):
            decoder_inputs.append(
                np.array([decoder_input[length_idx] for decoder_input in raw_decoder_input], dtype=np.int32))
            target_weights.append(np.array([
                0.0 if length_idx == output_seq_len - 1 or decoder_input[length_idx] == self.PAD_ID else 1.0 for
                decoder_input in raw_decoder_input
            ], dtype=np.float32))
        return encoder_inputs, decoder_inputs, target_weights

    def seq_to_encoder(self, input_seq):
        """从输入空格分隔的数字id串，转成预测用的encoder、decoder、target_weight等
        """
        input_seq_array = [int(v) for v in input_seq.split()]
        encoder_input = [self.PAD_ID] * (self.input_seq_len - len(input_seq_array)) + input_seq_array
        decoder_input = [self.GO_ID] + [self.PAD_ID] * (self.output_seq_len - 1)
        encoder_inputs = [np.array([v], dtype=np.int32) for v in encoder_input]
        decoder_inputs = [np.array([v], dtype=np.int32) for v in decoder_input]
        target_weights = [np.array([1.0], dtype=np.float32)] * self.output_seq_len
        return encoder_inputs, decoder_inputs, target_weights    
    
class Processor1(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def __init__(self):
        super(Processor, self).__init__()
        embedding_path = os.path.join(DATA_PATH, 'glove.txt')
        embeddings_index = {}
        with open(embedding_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float64')
                embeddings_index[word] = coefs
        self.w2v = embeddings_index
        word_path = os.path.join(DATA_PATH, 'vocab.json')
        with open(word_path, encoding='utf-8') as f:
            self.w2d = json.loads(f.read())
        self.max_sts_len = 512
        self.embedding_len = 100
       
    def input_x(self, text):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        vec = words2vec(text,self.w2v,self.embedding_len,self.max_sts_len)
        token = words2vec(text,self.w2d,1,self.max_sts_len)
        return vec,token

    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, stars):
        '''
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        '''
        if stars not in [1,2,3,4,5]:
            stars=2
        labels = int(stars-1)
        return labels

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        
        return np.argmax(data)+1

class NLPFlyAI(Dataset):
    def __init__(self, root, df, transform=None):
        self.root=root
        self.df=df
        with open(os.path.join(DATA_PATH, 'words.dict'), 'r') as f:
            js = f.read()
            self.word2id_dict = json.loads(js)
        self.id2word_dict = {}
        for k, v in self.word2id_dict.items():
            self.id2word_dict[v] = k
        self.length = len(self.word2id_dict) + 5
        # 空值填充0
        self.PAD_ID = 1
        # 输出序列起始标记
        self.GO_ID = 0
        # 结尾标记
        self.EOS_ID = 2
        self.UNK = 3
        self.input_seq_len = 10
        self.output_seq_len = 10
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        question,answer = self.df.iloc[index]    
        x_ids = self.get_id_list_from(question)
        cut = self.input_seq_len
        x_ids_tr = np.array(x_ids[:cut] if len(x_ids) > cut else x_ids+[self.PAD_ID]*(cut-len(x_ids)))
        y_ids = self.get_id_list_from(answer)
        cut = self.output_seq_len
        y_ids_tr = np.array(y_ids[:cut] if len(y_ids) > cut else y_ids+[self.PAD_ID]*(cut-len(y_ids)))
        return x_ids_tr,y_ids_tr

    def id2word(self, id):
        id = int(id)
        if id in self.id2word_dict:
            return self.id2word_dict[id]
        else:
            return ""

    def get_id_list_from(self, sentence):
        sentence_id_list = []
        seg_list = jieba.cut(sentence)
        for str in seg_list:
            sentence_id_list.append(self.word2id(str))
        return sentence_id_list

    def word2id(self, word):
        if not isinstance(word, str):
            print("Exception: error word not unicode")
            sys.exit(1)
        if word in self.word2id_dict:
            return self.word2id_dict[word]
        else:
            return 3  # UNK

    
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

