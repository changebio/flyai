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
from net import Net, LSTMNet
from path import MODEL_PATH
from processor import DQuestionFlyAI,load_data, make_weights_for_balanced_classes,BertFlyAI
from utils import Bunch
from pytorch_pretrained_bert import BertAdam
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM,BertForSequenceClassification

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

#settings
settings = {
'net':'Bert',
'nc':2,    
'lr': 0.0001,
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
trn,val = load_data()
train_dataset = BertFlyAI(root=DATA_PATH,df=trn)
val_dataset = BertFlyAI(root=DATA_PATH,df=val)
weight = make_weights_for_balanced_classes(trn.label,settings.nc)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weight,len(weight))
train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=args.BATCH, sampler =sampler, **kwargs)
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
#net = LSTMNet()
net = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
gpu = torch.cuda.is_available()
if gpu:
    net.cuda()
#optimize and loss
print("4.optimize and loss. learning rate %g" % settings.lr)
optimizer = BertAdam(net.parameters(), lr=settings.lr, weight_decay=1e-4)
#optimizer = Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))  
loss_fn = nn.CrossEntropyLoss()  

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
        #x1,x2, y_train = next(train_iter)
        x_input_ids,x_input_mask,x_segment_ids, y_train = next(train_iter)
        print(batch_idx,"data len",torch.stack(x_input_ids,dim=1).shape,len(y_train))
    except:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight,len(weight))
        train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.BATCH, sampler =sampler, **kwargs)
        batch_idx = 0
        train_iter = iter(train_loader)
        #x1,x2, y_train = next(train_iter)
        x_input_ids,x_input_mask,x_segment_ids, y_train = next(train_iter)
        print(len(x_input_ids),len(x_input_mask),len(y_train),"data len")

    if gpu:
        x_input_ids=torch.stack(x_input_ids,dim=1).to('cuda')
        x_input_mask=torch.stack(x_input_mask,dim=1).to('cuda')
        x_segment_ids = torch.stack(x_segment_ids,dim=1).to('cuda')
        y_train = y_train.to('cuda')


    outputs = net(x_input_ids,x_segment_ids,x_input_mask)
    optimizer.zero_grad()
    loss = loss_fn(outputs, y_train)
    loss.backward()
    optimizer.step()
    if batch_idx % settings.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(y_train), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

if settings.save_model:       
    model.save_model(net, MODEL_PATH, overwrite=True)
     
