# -*- coding: utf-8 -*
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
import os
from net import Decoder, Encoder, Net
from path import MODEL_PATH, LOG_PATH
from flyai.dataset import Dataset
from model import Model


# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
args = parser.parse_args()

# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

device = torch.device(device) 



os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# 数据获取辅助类
data = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=args.BATCH)
en=Encoder(20,64)
de=Decoder(3507,20,64)
network = Net(en,de,device)
loss_fn = nn.CrossEntropyLoss()

optimizer = Adam(network.parameters())

model = Model(data)
iteration = 0


lowest_loss = 10
# 得到训练和测试的数据
for i in range(data.get_step()):
    network.train()
    
    # 得到训练和测试的数据
    x_train, y_train = data.next_train_batch() # 读取数据; shape:(sen_len,batch,embedding)
    x_test, y_test = data.next_validation_batch() # 读取数据; shape:(sen_len,batch,embedding)

    batch_len = y_train.shape[0]

    input_lengths = x_train[:,-1,0]
    input_lengths = input_lengths.tolist()

    input_lengths = [int(x) for x in input_lengths]
    y_lengths = y_train[:,-1]
    y_lengths = y_lengths.tolist()
    
    x_train = x_train[:,:-1,:] ## 除去长度信息
    x_train = torch.from_numpy(x_train) #shape:(batch,sen_len,embedding)
    x_train = x_train.float().to(device) 
    y_train = y_train[:,:-1] ## 除去长度信息
    y_train = torch.from_numpy(y_train) #shape:(batch,sen_len)
    y_train = torch.LongTensor(y_train)
    y_train = y_train.to(device) 

    seq_pairs = sorted(zip(x_train.contiguous(), y_train.contiguous(), input_lengths, y_lengths), key=lambda x: x[2], reverse=True)
    x_train, y_train, input_lengths, y_lengths = zip(*seq_pairs)
    x_train = torch.stack(x_train,dim=0).permute(1,0,2).contiguous()
    y_train = torch.stack(y_train,dim=0).permute(1,0).contiguous()
    input_length_tensor = torch.tensor(input_lengths).long()
    y_lengths = torch.tensor(y_lengths).long()

    outputs = network(x_train,input_lengths,y_train)

    optimizer.zero_grad()
    outputs = outputs.float()

    loss = loss_fn(outputs.view(-1, outputs.shape[2]), y_train.view(-1))

    # backward transmit loss
    loss.backward()
    # adjust parameters using Adam
    optimizer.step()
    print(loss)

    if loss < lowest_loss:
        lowest_loss = loss
        model.save_model(network, MODEL_PATH, overwrite=True)
        print("step %d, best lowest_loss %g" % (i, lowest_loss))
    print(str(i) + "/" + str(data.get_step()))
    

