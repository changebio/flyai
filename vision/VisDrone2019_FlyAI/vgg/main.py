# -*- coding: utf-8 -*
# author Huangyin

from __future__ import print_function
import argparse

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.optim import Adam
from torch.autograd import Variable

from flyai.dataset import Dataset
from flyai.source.base import DATA_PATH

from model import Model
from path import MODEL_PATH
from processor import ImageLabel,make_weights_for_balanced_classes
from utils import Bunch,load_data
trn,val = load_data()
from vgg import VGG
from net import Net

    
def pretrained_net(net,class_num):
    if net=='densenet121':
        cnn = torchvision.models.densenet121(pretrained=True)
        for param in cnn.parameters():
            param.requires_grad = False
        num_features = cnn.classifier.in_features
        cnn.classifier = nn.Sequential(nn.Linear(num_features, class_num),nn.Sigmoid())
    elif net=='resnet50':
        cnn = torchvision.models.resnet50(pretrained=False)
        for param in cnn.parameters():
            param.requires_grad = True
        num_features = cnn.fc.in_features
        cnn.fc = nn.Sequential(nn.Linear(num_features, class_num),nn.Sigmoid())
    elif net=='rcnn':
        cnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
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
       
def fit(train_dl,model,loss_func,opt,epochs=1,gpu=True):
    n=0
    for epoch in range(epochs):
        for xb,yb in train_dl:
            #print(xb.shape,yb.shape)
            if gpu:
                xb = Variable(xb.cuda())
                yb = Variable(yb.float().cuda())
            
            try:
                pred = model(xb)
                loss = loss_func(pred,yb)
                loss.backward()
                opt.step()
                opt.zero_grad()
            except:
                n+=1
                #model.eval()
                #pred = model(xb)
                #print(pred)
    print(n)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=4, type=int, help="batch size")
    args = parser.parse_args()
    
  
    #settings
    settings = {
    'net':'rcnn',
    'sigmoid':True,
    'nc':18,    
    'lr': 0.001,
    'seed': 1,
    'log_interval': 100,
    'save_model': True,
    'predict': True}
    print("1.settings",settings)
    settings = Bunch(settings)
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(settings.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    
    #load data
    data = Dataset()
    model = Model(data)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    train_transforms= transforms.Compose([
        #transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        #transforms.Resize((224,224)),
        transforms.ToTensor(),
        #normalize,
    ])
    val_transforms = transforms.Compose([
        #transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #normalize,
    ])
    train_dataset = ImageLabel(root=DATA_PATH,df=trn,transform=train_transforms)
    val_dataset = ImageLabel(root=DATA_PATH,df=val,transform=val_transforms)
    #weight = make_weights_for_balanced_classes(trn.label,settings.nc)
    #weight = torch.DoubleTensor(weight)
    #print('weight',len(weight),weight[:10])
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.BATCH, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.BATCH, **kwargs)
    n_train = len(train_dataset)
    batch_train = n_train/args.BATCH
    print(args.BATCH,n_train/args.BATCH)
    n_test = len(val_dataset)
    batch_test = n_test/args.BATCH
    print("2. load data. train_dataset %d,batch %d, val_dataset %d, batch %d." % (n_train,batch_train,n_test,batch_test))
  
    #load net structure
    print("3.load net structure: %s, number of class: %d" % (settings.net,settings.nc))
    #net = VGG(settings.net,settings.nc,settings.sigmoid)
    net = pretrained_net(settings.net,settings.nc)
    #net = Net()
    gpu = torch.cuda.is_available()
    if gpu:
        net.cuda()
    #optimize and loss
    print("4.optimize and loss. learning rate %g" % settings.lr)
    optimizer = Adam(net.parameters(), lr=settings.lr, betas=(0.9, 0.999)) 
    #optimizer = Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))  
    #loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss()  # 定义损失函数

    #train and test
    print("5.***************train and test*********************")
    best_accuracy = 0
    #train_iter = iter(train_loader)
    #batch_idx = 0
    #losses=[]
    net.train()
    fit(train_loader,net,loss_fn,optimizer,args.EPOCHS)
    
        
    if settings.save_model:       
        model.save_model(net, MODEL_PATH, overwrite=True)
      
        
    if settings.predict:
        #print("6******prediction*****",x_train.shape,y_train.shape)
        import random
        _,_,x_test,y_text = data.get_all_data()
        test_idx = random.sample(range(len(x_test)),10)
        print(model.predict_all([x_test[i] for i in test_idx]),[y_text[i] for i in test_idx])
        #print('Losses',losses)
       