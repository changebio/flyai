import json
import numpy as np
import config
import tensorflow as tf

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

from flyai.processor.base import Base
from flyai.processor.download import check_download
from flyai.utils.yaml_helper import Yaml
from flyai.utils import read_data

from path import DATA_PATH
from pathlib import Path

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)
        
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
    p = Path(sys.path[0])
    print([x for x in p.iterdir()])
    print([x for x in (p/'data').iterdir()])
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


def load_word2vec_embedding(vocab_size):
    '''
        加载外接的词向量。
        :return:
    '''
    print('loading word embedding, it will take few minutes...')
    embeddings = np.random.uniform(-1,1,(vocab_size + 2, config.embeddings_size))
    # 保证每次随机出来的数一样。
    rng = np.random.RandomState(23455)
    unknown = np.asarray(rng.normal(size=(config.embeddings_size)))
    padding = np.asarray(rng.normal(size=(config.embeddings_size)))
    f = open(config.word_embedding_file)
    embedding_table = json.load(f)

    with open(config.src_vocab_file,'r') as fw:
        words_dic=json.load(fw)
    for index, line in enumerate(embedding_table):
        values = embedding_table[line]
        try:
            coefs = np.asarray(values, dtype='float32')  # 取向量
        except ValueError:
            # 如果真的这个词出现在了训练数据里，这么做就会有潜在的bug。那coefs的值就是上一轮的值。
            print(line, values)
        if line in words_dic:
            embeddings[int(words_dic[line])] = coefs   # 将词和对应的向量存到字典里
    f.close()
    # 顺序不能错，这个和unkown_id和padding id需要一一对应。
    embeddings[-2] = unknown
    embeddings[-1] = padding

    return tf.get_variable("embeddings", dtype=tf.float32,
                           shape=[vocab_size + 2, config.embeddings_size],
                           initializer=tf.constant_initializer(embeddings), trainable=False)