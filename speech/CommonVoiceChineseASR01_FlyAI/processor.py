# -*- coding: utf-8 -*
from flyai.processor.download import check_download
from flyai.processor.base import Base
from path import DATA_PATH  # 导入输入数据的地址
import os
import json
import numpy as np
import librosa


x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗



class Processor(Base):
    # 该参数需要与app.yaml的Model的input-->columns->name 一一对应
    def __init__(self):
        with open(os.path.join(DATA_PATH, 'words.json')) as fin:
            words = json.loads(fin.read())
        words = list(words.keys())
        words = [" ", "<unk>"] + words
        self.max_audio_len = 900
        self.max_tgt_len = 38
        self.char_dict = dict()
        self.char_dict_res = dict()
        for i, word in enumerate(words):
            self.char_dict[word] = i
            self.char_dict_res[i] = word
        
              
    def input_x(self, audio_path):
        '''
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        '''
        try:
            path = check_download(audio_path, DATA_PATH)
            wav, sr = librosa.load(path, mono=True)
            mfcc = librosa.feature.mfcc(wav, sr, hop_length=int(0.010 * sr), n_fft=int(0.025 * sr))
            mfcc = mfcc.transpose((1,0))
        except Exception as e:
            print('mfcc error')
        try:
        
            if len(mfcc) >= self.max_audio_len:
                mfcc = mfcc[:self.max_audio_len]
                origanal_len = self.max_audio_len
            else:
                origanal_len = len(mfcc)
                mfcc = np.concatenate((mfcc,np.zeros([self.max_audio_len-origanal_len,20])),0)
            mfcc = np.concatenate((mfcc,np.array([origanal_len for j in range(20)]).reshape([1,20]))) ## 最后一行元素为句子实际长度

        except:
            print('conc error')
        return mfcc


    # 该参数需要与app.yaml的Model的output-->columns->name 一一对应
    def input_y(self, label):
        word_list = []
        for word in label:
            word_idx = self.char_dict.get(word)
            if word_idx is not None:
                word_list.append(word_idx)
                
        origanal_len = len(word_list)
        if len(word_list) >= self.max_tgt_len:
            origanal_len = self.max_tgt_len
            word_list = word_list[:self.max_tgt_len]
        else:
            origanal_len = len(word_list)
            for i in range(len(word_list), self.max_tgt_len):
                word_list.append(0) ## 不够长度则补0  
        word_list.append(origanal_len) ##最后一个元素为句子长度
        return word_list

    def output_y(self, data):
        '''
        验证时使用，把模型输出的y转为对应的结果
        '''
        output_words = [self.char_dict_res[np.argmax(word_prob)] for word_prob in data]
        return output_words
