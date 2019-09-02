# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import argparse
import tensorflow as tf
from flyai.dataset import Dataset
from model import Model
from path import MODEL_PATH
import os
import numpy as np

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
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)

# 实现自己模型
model = Model(dataset)

'''
dataset.get_step() 获取数据的总迭代次数
'''

best_score = 0
learning_rate = 0.001
loss_mean = np.inf

for step in range(dataset.get_step()):
    x_train, y_train = dataset.next_train_batch()
    x_val, y_val = dataset.next_validation_batch()
    _, loss = model.session.run([model.train_op, model.get_loss], feed_dict={model.x: x_train, model.y: y_train, model.lr: learning_rate})
    print('step: %d/%d, loss: %.2f' %(step,dataset.get_step(), loss))
    if(step % 10 ==0):
        if(loss < loss_mean):
            loss_mean = loss
            model.saver.save(model.session, os.path.join(MODEL_PATH, 'best'), global_step=step)
            print('保存模型！！！！！！！！！！！')

    # if(step %10 == 0):
    #     model.session.run(model.vgg.conv_final, feed_dict={model.x: x_val})
    #
    #     '''
    #     实现自己的保存模型逻辑
    #     '''
    #     model.save_model(sess, MODEL_PATH, overwrite=True)
    #     print(str(step + 1) + "/" + str(dataset.get_step()))


import random
_,_,x_test,y_text = dataset.get_all_data()
test_idx = random.sample(range(len(x_test)),10)
print(model.predict_all([x_test[i] for i in test_idx]),[y_text[i] for i in test_idx])

model.close_session()
print('训练结束！！！')
'''
本项目说明：
1、这个样例只提供简单流程，具体细节部分没有实现。
2、在评测时，只需参考 model中的 decode_pred_bb 方法。
   对于一张图片，只需预测的所有类别图片按照得分排序
    <class_name> <left> <top> <right> <bottom>

'''

