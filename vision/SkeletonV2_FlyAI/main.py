# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
from flyai.dataset import Dataset
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Conv3D, MaxPool3D
from keras.layers.normalization import BatchNormalization
from model import Model
from path import MODEL_PATH

'''
样例代码仅供参考学习，可以自己修改实现逻辑。
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=5, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

'''
实现自己的网络结构
'''
seque = Sequential()
chanDim = -1

seque.add(Conv3D(filters=32, kernel_size=(3, 3, 3), padding='Same', activation='relu', input_shape=(50,64,64,1)))
seque.add(BatchNormalization(axis=chanDim))
seque.add(MaxPool3D(pool_size=(3, 3, 3)))
seque.add(Dropout(0.5))
seque.add(Flatten())
seque.add(Dense(1024, activation='relu'))
seque.add(BatchNormalization())
seque.add(Dropout(0.5))
seque.add(Dense(120, activation='softmax'))

seque.summary()

seque.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

'''
dataset.get_step() 获取数据的总迭代次数

'''
best_score = 0
for step in range(dataset.get_step()):
    x_train, y_train = dataset.next_train_batch()
    x_val, y_val = dataset.next_validation_batch()
    history = seque.fit(x_train, y_train,
                        batch_size=args.BATCH,
                        verbose=1)
    score = seque.evaluate(x_val, y_val, verbose=0)
    if score[1] > best_score:
        best_score = score[1]
        '''
        保存模型
        '''
        model.save_model(seque, MODEL_PATH, overwrite=True)
        print("step %d, best accuracy %g" % (step, best_score))
    print(str(step + 1) + "/" + str(dataset.get_step()))
