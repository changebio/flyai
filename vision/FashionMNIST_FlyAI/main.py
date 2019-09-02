# -*- coding: utf-8 -*
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import torchvision

from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

def pretrained_net(net,class_num):
    if net=='densenet121':
        cnn = torchvision.models.densenet121(pretrained=True)
        for param in cnn.parameters():
            param.requires_grad = False
        num_features = cnn.classifier.in_features
        cnn.classifier = nn.Linear(num_features, class_num)
    elif net=='resnet34':
        cnn = torchvision.models.resnet34(pretrained=True)
        for param in cnn.parameters():
            param.requires_grad = False
        num_features = cnn.fc.in_features
        cnn.fc = nn.Sequential(nn.Linear(num_features, class_num),nn.Sigmoid())
    return cnn

def score(p,y):
    _,yp = torch.max(p.data,1)
    return (yp == y).sum().item()/len(y)

    
def eval(model, x_test, y_test):
    net.eval()
    batch_eval = model.batch_iter(x_test, y_test)
    total_acc = 0.0
    data_len = len(x_test)
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        outputs = net(x_batch)
        _, prediction = torch.max(outputs.data, 1)
        correct = (prediction == y_batch).sum().item()
        acc = correct / batch_len
        total_acc += acc * batch_len
    return total_acc / data_len



parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

#load data
data = Dataset()
model = Model(data)
#load net structure
#net = pretrained_net(args.MODEL,args.CLASS_NUM)
net = pretrained_net('densenet121',10)
gpu = torch.cuda.is_available()
if gpu:
    net.cuda()
#optimize and loss
optimizer = Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))  
loss_fn = nn.CrossEntropyLoss()  

#train and test
best_accuracy = 0
for i in range(args.EPOCHS):
    net.train()
    x_train, y_train, x_test, y_test = data.next_batch(args.BATCH)  
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_train = x_train.float()
    if gpu:
        x_train = Variable(x_train.cuda())
        y_train = Variable(y_train.cuda())


    outputs = net(x_train)
    optimizer.zero_grad()
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print("train loss: %g, train score: %g" % (loss.data.item(), score(outputs,y_train)))

        x_test = torch.from_numpy(x_test)
        y_test = torch.from_numpy(y_test)
        x_test = x_test.float()
        if gpu:
            x_test = Variable(x_test.cuda())
            y_test = Variable(y_test.cuda())
        y_outs = net(x_test)
        loss = loss_fn(y_outs, y_test)
        print('test loss: %g, test score: %g' % (loss.data.item(),score(y_outs,y_test)))

        train_accuracy = eval(model, x_test, y_test)
        if train_accuracy > best_accuracy:
            best_accuracy = train_accuracy
            model.save_model(net, MODEL_PATH, overwrite=True)
            print("step %d, best accuracy %g" % (i, best_accuracy))





