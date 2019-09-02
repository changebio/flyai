# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017
@author: user
"""
import argparse
from flyai.dataset import Dataset
from model import Model
from path import MODEL_PATH
import tensorflow as tf
import os

'''
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''


'''
项目中的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=1, type=int, help="batch size")
args = parser.parse_args()
img_shape = [224, 224, 3]
num_classes = 9

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)
'''
实现自己的网络机构
'''

model.my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

'''
dataset.get_step() 获取数据的总迭代次数

'''

print('all step is : %d'%(dataset.get_step()))
print('all train len is : %d'%(dataset.get_train_length()))
print('all validation len is : %d'%(dataset.get_validation_length()))
print('batch_size is %d, epoch_size is %d'%(args.BATCH, args.EPOCHS))
best_score = 0
for step in range(dataset.get_step()):
    print('-----------------step %d/%d' %(step, dataset.get_step()))
    x_train, y_train = dataset.next_train_batch()
    x_test, y_test = dataset.next_validation_batch()
    history = model.my_model.fit(x_train, y_train, epochs=1, batch_size=args.BATCH)

    loss, score = model.my_model.evaluate(x_test, y_test, verbose=0)
    print('now val loss is %f, score is %f'%(loss, score))
    if score > best_score:
        print('saved score :%f !!!!!!!'%score)
        best_score = score
        model.save_model(model.my_model, MODEL_PATH)
