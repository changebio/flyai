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

def eval(model, x_test, y_test):
    cnn.eval()
    batch_eval = model.batch_iter(x_test, y_test)
    total_acc = 0.0
    data_len = len(x_test)
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        outputs = cnn(x_batch)
        _, prediction = torch.max(outputs.data, 1)
        correct = (prediction == y_batch).sum().item()
        acc = correct / batch_len
        total_acc += acc * batch_len
    return total_acc / data_len

cnn = Net().to(device)
optimizer = Adam(cnn.parameters(), lr=0.001, betas=(0.9, 0.999))  # 选用AdamOptimizer
loss_fn = nn.CrossEntropyLoss()  # 定义损失函数

# 训练并评估模型

data = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(data)

loss_list = []
acc_list = []
best_accuracy = 0
for i in range(dataset.get_step()):
    cnn.train()
    x_train, y_train = dataset.next_train_batch()
    x_test, y_test = dataset.next_validation_batch()

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_train = x_train.float().to(device)
    y_train = y_train.long().to(device)

    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    x_test = x_test.float().to(device)
    y_test = y_test.long().to(device)
    
    outputs = cnn(x_train)
    _, prediction = torch.max(outputs.data, 1)
    print(prediction)

    optimizer.zero_grad()
    #print(x_train.shape,outputs.shape,y_train.shape)
    loss = loss_fn(outputs, y_train)
    loss_list.append(loss.item())
    loss.backward()
    optimizer.step()
    print(loss)
    # 若测试准确率高于当前最高准确率，则保存模型
    train_accuracy = eval(model, x_test, y_test)
    acc_list.append(train_accuracy)
    if train_accuracy > best_accuracy:
        best_accuracy = train_accuracy
        model.save_model(cnn, MODEL_PATH, overwrite=True)
        print("step %d, best accuracy %g" % (i, best_accuracy))

    print(str(i) + "/" + str(args.EPOCHS))
