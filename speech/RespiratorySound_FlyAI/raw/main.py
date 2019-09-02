# -*- coding: utf-8 -*-
"""
Created on Fir May 17 19:44:02 2019

@author: M
"""

import argparse
import codecs
import sys
import os.path
import tensorflow as tf

from flyai.dataset import Dataset
from model import Model
from keras.backend.tensorflow_backend import set_session

from utils import Bunch,load_data
trn,val = load_data()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

FLAGS = None

feat_type = 'mfcc'  ##'dmfcc'#None  'normmfcc'
label_smooth = False

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


'''
项目的超参
'''
parser = argparse.ArgumentParser()
# 111902 约等于512*200
parser.add_argument("-e", "--EPOCHS", default=200, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=4, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model_helper = Model(dataset, do_train=True)

'''
实现自己的网络结构
'''

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.InteractiveSession(config=config)
    set_session(sess)
    cnn_model = model_helper.model
    cnn_model.train_model(sess, dataset, model_helper, args.BATCH, args.EPOCHS)

if __name__ == '__main__':
    main()