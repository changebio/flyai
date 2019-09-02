# -*- coding: utf-8 -*
# author Huangyin

from __future__ import print_function
import argparse
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
from processor import SLR1FlyAI
from utils import Bunch

from net import SimpleNet
    
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
    elif net=='selfdef':
        cnn = SimpleNet()
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


#settings
settings = {
'net':'selfdef',
'nc':500,    
'lr': 0.001,
'seed': 1,
'log_interval': 100,
'save_model': True}
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
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
train_dataset = SLR1FlyAI(root=DATA_PATH,df=data.db.source.data)
val_dataset = SLR1FlyAI(root=DATA_PATH,df=data.db.source.test)
train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=args.BATCH, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(val_dataset,
    batch_size=args.BATCH, shuffle=True, **kwargs)
n_train = len(train_dataset)
batch_train = n_train/args.BATCH
print(args.BATCH,n_train/args.BATCH)
n_test = len(val_dataset)
batch_test = n_test/args.BATCH
print("2. load data. train_dataset %d,batch %d, val_dataset %d, batch %d." % (n_train,batch_train,n_test,batch_test))

#load net structure
print("3.load net structure: %s, number of class: %d" % (settings.net,settings.nc))
net = pretrained_net(settings.net,settings.nc)
gpu = torch.cuda.is_available()
if gpu:
    net.cuda()
#optimize and loss
print("4.optimize and loss. learning rate %g" % settings.lr)
optimizer = Adam(net.parameters(), lr=settings.lr, betas=(0.9, 0.999)) 
#optimizer = Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))  
loss_fn = nn.CrossEntropyLoss()  

#train and test
print("5.***************train and test*********************")
best_accuracy = 0
train_iter = iter(train_loader)
batch_idx = 0
for i in range(args.EPOCHS):
    #net.train()
    #x_train, y_train, x_test, y_test = data.next_batch(args.BATCH) 
    #x_train = torch.from_numpy(x_train)
    #y_train = torch.from_numpy(y_train)
    #x_train = x_train.float()
    try:
        batch_idx +=1
        x_train, y_train = next(train_iter)
        print(batch_idx,"data len",len(x_train),len(y_train))
    except:
        batch_idx = 0
        train_iter = iter(train_loader)
        x_train, y_train = next(train_iter)
        print(len(x_train),len(y_train),"data len")
    #x_train = torch.from_numpy(x_train)
    #y_train = torch.from_numpy(y_train)
    if gpu:
        x_train = Variable(x_train.cuda())
        y_train = Variable(y_train.cuda())


    outputs = net(x_train)
    optimizer.zero_grad()
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()
    if batch_idx % settings.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            i, batch_idx * len(x_train), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

if settings.save_model:       
    model.save_model(net, MODEL_PATH, overwrite=True)
