# -*- coding: utf-8 -*
import argparse
import torch
import torch.nn as nn
from net import Net
from flyai.dataset import Dataset
from torch.autograd import Variable
from torch.optim import Adam

from model import Model
from path import MODEL_PATH

# 数据获取辅助类
dataset = Dataset()

# 模型操作辅助类
model = Model(dataset)

# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=100, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=24, type=int, help="batch size")
args = parser.parse_args()

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
#device = 'cpu'    
device = torch.device(device) 

cnn = Net().to(device)
optimizer = Adam(cnn.parameters(), lr=0.001, betas=(0.9, 0.999))  # 选用AdamOptimizer
loss_fn = nn.SmoothL1Loss()  # 定义损失函数

# 训练并评估模型

data = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=args.BATCH)
model = Model(data)

lowest_loss = 1e5
for i in range(data.get_step()):
    cnn.train()
    x_train, y_train = data.next_train_batch() 
    x_test, y_test = data.next_validation_batch()

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_train = x_train.float().to(device)
    y_train = y_train.float().to(device)
    
    outputs = cnn(x_train)
    #_, prediction = torch.max(outputs.data, 1)

    optimizer.zero_grad()
    #print(x_train.shape,outputs.shape,y_train.shape)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(loss)

    if loss < lowest_loss:
        lowest_loss = loss
        model.save_model(cnn, MODEL_PATH, overwrite=True)
        print("step %d, lowest loss %g" % (i, lowest_loss))

    print(str(i) + "/" + str(data.get_step()))
