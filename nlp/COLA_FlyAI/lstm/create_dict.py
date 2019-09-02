# -*- coding: utf-8 -*
import jieba
import json
import os
import pandas

from path import DATA_PATH

# 不加载自定义词库62048，加载后61646

special_chars = ['_PAD_', '_EOS_', '_SOS_', '_UNK_']  # '_START_'
_PAD_ = 0
_EOS_ = 1
_UNK_ = 2
_SOS_ = 3


# _START_ = 3
def create(filename, DICT_PATH, LABEL_PATH):
    print('save to', DICT_PATH, LABEL_PATH)
    word_dict = dict()
    for i, word in enumerate(special_chars):
        word_dict[word] = i
    label_dict = dict()
    f = pandas.read_csv(filename, usecols=['text', 'gid'])
    if LABEL_PATH is not None:
        labels = f.values[:, 1]
        labels = labels.astype('str')
        for label in labels:
            if label not in label_dict:
                label_dict[label] = len(label_dict)

    data = f.values[:, 0]
    data = data.astype('str')
    for sentence in data:
        # keep_len = 1000
        # if len(sentence)>keep_len:
        #     sentence = sentence[:keep_len]
        terms = jieba.cut(sentence, cut_all=False)
        # sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。；？、~@#￥%……&*（）]+", " ", sentence)
        for c in terms:
            if c not in word_dict:
                word_dict[c] = len(word_dict)
        for c in sentence:
            if c not in word_dict:
                word_dict[c] = len(word_dict)
    with open(os.path.join(DICT_PATH), 'w', encoding='utf-8') as fout:
        json.dump(word_dict, fout)
    if LABEL_PATH is not None:
        with open(os.path.join(LABEL_PATH), 'w', encoding='utf-8') as fout:
            json.dump(label_dict, fout)
    print('build dict done.')


def load_dict():
    char_dict_re = dict()
    dict_path = os.path.join(DATA_PATH, 'words.dict')
    with open(dict_path, encoding='utf-8') as fin:
        char_dict = json.load(fin)
    for k, v in char_dict.items():
        char_dict_re[v] = k
    return char_dict, char_dict_re


def load_label_dict():
    char_dict_re = dict()
    dict_path = os.path.join(DATA_PATH, 'label.dict')
    with open(dict_path, encoding='utf-8') as fin:
        char_dict = json.load(fin)
    for k, v in char_dict.items():
        char_dict_re[v] = k
    return char_dict, char_dict_re
