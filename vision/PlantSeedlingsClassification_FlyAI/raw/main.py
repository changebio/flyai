# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

import argparse
import torch
import torch.nn as nn

from flyai.dataset import Dataset
from model import Model
from net import Net
from path import MODEL_PATH
from torch.optim import Adam

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

'''
实现自己的网络机构
'''
# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)
net = Net().to(device)

optimizer = Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))  # 选用AdamOptimizer
loss_fn = nn.CrossEntropyLoss()  # 定义损失函数

'''
dataset.get_step() 获取数据的总迭代次数

'''
best_score = 0
for step in range(dataset.get_step()):
    print("Step:{}/{}".format(step + 1, dataset.get_step()))

    net.train()

    x_train, y_train = dataset.next_train_batch()
    x_val, y_val = dataset.next_validation_batch()

    batch_len = len(x_val)

    x_ = torch.from_numpy(x_train).float().to(device)
    y_ = torch.from_numpy(y_train).to(device, dtype=torch.int64)
    x_val_ = torch.from_numpy(x_val).float().to(device)
    y_val_ = torch.from_numpy(y_val).to(device, dtype=torch.int64)

    outputs = net(x_)
    optimizer.zero_grad()
    loss = loss_fn(outputs, torch.max(y_, 1)[1])
    loss.backward()
    optimizer.step()
    print("loss detach: {}".format(loss.detach()))

    '''
    实现自己的模型保存逻辑
    '''
    outputs_val = net(x_val_)
    _, prediction = torch.max(outputs_val.data, 1)
    correct = (prediction == torch.max(y_val_, 1)[1]).sum().item()
    accuracy = correct / batch_len

    print("{}step(s) loss :{} acc:{}".format(step + 1, loss.detach(), accuracy))
    if accuracy > best_score:
        best_score = accuracy
        model.save_model(net, MODEL_PATH, overwrite=True)

        print("Step: {}, best accuracy is {}. save model.".format(step, best_score))

    print(str(step + 1) + "/" + str(dataset.get_step()))

print("The best score is : {}".format(best_score))
