# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import time

import argparse
from flyai.dataset import Dataset
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential

from model import Model
from path import MODEL_PATH
from utils import Bunch,load_data
trn,val = load_data()

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=8, type=int, help="batch size")
args = parser.parse_args()
sqeue = Sequential()

# 第一个卷积层，32个卷积核，大小５x5，卷积模式SAME,激活函数relu,输入张量的大小
sqeue.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(224, 224, 3)))
sqeue.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
# 池化层,池化核大小２x2
sqeue.add(MaxPool2D(pool_size=(2, 2)))
sqeue.add(Dropout(0.25))
sqeue.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
sqeue.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
sqeue.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
sqeue.add(Dropout(0.25))
# 全连接层,展开操作，
sqeue.add(Flatten())
# 添加隐藏层神经元的数量和激活函数
sqeue.add(Dense(256, activation='relu'))
sqeue.add(Dropout(0.25))
# 输出层
sqeue.add(Dense(2, activation='softmax'))

# 输出模型的整体信息
sqeue.summary()

sqeue.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

dataset = Dataset()
model = Model(dataset)
best_score = 0
for epochs in range(args.EPOCHS):
    first_time = int(time.time())
    x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)
    history = sqeue.fit(x_train, y_train,
                        batch_size=args.BATCH,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = sqeue.evaluate(x_test, y_test, verbose=0)
    if score[1] > best_score:
        best_score = score[1]
        model.save_model(sqeue, MODEL_PATH, overwrite=True)
        print("step %d, best accuracy %g" % (epochs, best_score))
    print(str(epochs + 1) + "/" + str(args.EPOCHS))
