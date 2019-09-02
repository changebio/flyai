# -*- coding: utf-8 -*
# author ChangeBio

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torch.autograd import Variable

from flyai.dataset import Dataset
from flyai.source.base import DATA_PATH

from model import Model
from net import Net, LSTMNet,FCNet,Seq2SeqRNN
from path import MODEL_PATH
from processor import NLPFlyAI,load_data
from utils import Bunch
from torch.nn import functional as F

def eval(model, x_test, y_test):
    network.eval()
    total_acc = 0.0
    data_len = len(x_test[0])
    x1, x2 = x_test
    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    x1 = x1.float().to(device)
    x2 = x2.float().to(device)
    y_test = torch.from_numpy(y_test)
    y_test = y_test.to(device)
    batch_eval = model.batch_iter(x1, x2, y_test)

    for x_batch1, x_batch2, y_batch in batch_eval:
        outputs = network(x_batch1, x_batch2)
        _, prediction = torch.max(outputs.data, 1)
        correct = (prediction == y_batch).sum().item()
        total_acc += correct
    return total_acc / data_len

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
args = parser.parse_args()
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
model = Model(dataset)

#settings
settings = {
'net':'lstm',
'nc':5,    
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
trn,val = load_data()
train_dataset = NLPFlyAI(root=DATA_PATH,df=trn)
val_dataset = NLPFlyAI(root=DATA_PATH,df=val)
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
from create_dict import load_dict
stoi_dict,itos_dict  = load_dict(file="words.dict")
itos = [itos_dict[i] for i in range(len(itos_dict))]
from path import DATA_PATH
import os
import json
embedding_path = os.path.join(DATA_PATH, 'embedding.json')
with open(embedding_path) as f:
    ch_vecs = json.loads(f.read())
import numpy as np
new_ch_vecs = dict()
for k,v in ch_vecs.items():
    new_ch_vecs[k] = np.array([float(i) for i in v])
net = net = Seq2SeqRNN(new_ch_vecs,itos,200,new_ch_vecs,itos,200,10)
gpu = torch.cuda.is_available()
if gpu:
    net.cuda()
#optimize and loss
print("4.optimize and loss. learning rate %g" % settings.lr)
optimizer = Adam(net.parameters(), lr=settings.lr, weight_decay=1e-4)
#optimizer = Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))  
#loss_fn = nn.CrossEntropyLoss()  
def seq2seq_loss(input, target,weight=False):
    sl,bs = target.size()
    sl_in,bs_in,nc = input.size()
    if sl>sl_in: input = F.pad(input, (0,0,0,0,0,sl-sl_in))
    input = input[:sl]
    if weight:
        tar = target.view(-1)
    return F.cross_entropy(input.view(-1,nc), target.view(-1))#, ignore_index=1)


#train and test
print("5.***************train and test*********************")
best_accuracy = 0
train_iter = iter(train_loader)
batch_idx = 0
for epoch in range(args.EPOCHS):
    net.train()
    #x_train, y_train, x_test, y_test = data.next_batch(args.BATCH)  # 读取数据
    #batch_len = y_train.shape[0]
    #x1, x2 = x_train
    #x1 = torch.from_numpy(x1)
    #x2 = torch.from_numpy(x2)
    #x1 = x1.float().to(device)
    #x2 = x2.float().to(device)
    #y_train = torch.from_numpy(y_train)
    #y_train = y_train.to(device)
    try:
        batch_idx +=1
        x1, y_train = next(train_iter)
        #print(batch_idx,"data len",x1.shape,x2.shape,len(y_train))
    except:
        batch_idx = 0
        train_iter = iter(train_loader)
        x1, y_train = next(train_iter)
        print(len(x1),len(y_train),"data len")

    if gpu:
        x1 = Variable(x1.transpose(1,0).cuda())
        #x2 = Variable(x2.cuda())
        #x3 = Variable(x3.cuda())
        y_train = Variable(y_train.transpose(1,0).cuda())


    outputs = net(x1)
    optimizer.zero_grad()
    loss = seq2seq_loss(outputs, y_train)
    loss.backward()
    optimizer.step()
    if batch_idx % settings.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(y_train), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

if settings.save_model:       
    model.save_model(net, MODEL_PATH, overwrite=True)

if settings.predict:
    print(x1,y_train)
    import random
    _,_,x_test,y_text = dataset.get_all_data()
    test_idx = random.sample(range(len(x_test)),10)
    print(model.predict_all([x_test[i] for i in test_idx]),[y_text[i] for i in test_idx])
