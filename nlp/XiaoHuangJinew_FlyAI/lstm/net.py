# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
num_dims = 100


class Seq2SeqRNN(nn.Module):
    def __init__(self, vecs_enc, itos_enc, em_sz_enc, vecs_dec, itos_dec, em_sz_dec, out_sl,nh=256, nl=2):
        super().__init__()
        self.nl,self.nh,self.out_sl = nl,nh,out_sl
        self.emb_enc = create_emb(vecs_enc, itos_enc, em_sz_enc)
        self.emb_enc_drop = nn.Dropout(0.15)
        self.gru_enc = nn.GRU(em_sz_enc, nh, num_layers=nl, dropout=0.25)
        self.out_enc = nn.Linear(nh, em_sz_dec, bias=False)
        
        self.emb_dec = create_emb(vecs_dec, itos_dec, em_sz_dec)
        self.gru_dec = nn.GRU(em_sz_dec, em_sz_dec, num_layers=nl, dropout=0.1)
        self.out_drop = nn.Dropout(0.35)
        self.out = nn.Linear(em_sz_dec, len(itos_dec))
        self.out.weight.data = self.emb_dec.weight.data
        
    def forward(self, inp):
        sl,bs = inp.size()
        h = self.initHidden(bs)
        emb = self.emb_enc_drop(self.emb_enc(inp))
        enc_out, h = self.gru_enc(emb, h)
        h = self.out_enc(h)
      
        dec_inp = Variable(torch.zeros(bs).long().cuda())
        res = []
        for i in range(self.out_sl):
            emb = self.emb_dec(dec_inp).unsqueeze(0)
            outp, h = self.gru_dec(emb, h)
            outp = self.out(self.out_drop(outp[0]))
            res.append(outp)
            dec_inp = Variable(outp.data.max(1)[1].cuda())
            if (dec_inp==1).all(): break
        return torch.stack(res)
    
    def initHidden(self, bs): return Variable(torch.zeros(self.nl, bs, self.nh).cuda())

def create_emb(vecs, itos, em_sz):
    emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
    wgts = emb.weight.data
    miss = []
    for i,w in enumerate(itos):
        try: 
            wgts[i] = torch.from_numpy(vecs[w])
            #print(i,w,vecs[w][:5])
        except: 
            miss.append(w)
    print(len(miss),miss[5:10])
    return emb


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # input shape: 词向量维度，hidden个数，lstm层数
        self.LSTM_stack = nn.LSTM(num_dims, 64, num_layers=2, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.relu1 = nn.ReLU(True)
        self.fc1 = nn.Linear(10 * 64, 128)  ##  (max sentence length * hidden layer, 512)
        self.relu2 = nn.ReLU(True)
        self.dp = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x1, x2):
        x1, _ = self.LSTM_stack(x1.float())  # (batch, sentence_len, hidden_units)
        x2, _ = self.LSTM_stack(x2.float())
        x = x1 * x2

        # use every word in the sentence
        x = x.contiguous().view(-1, x.size(1) * x.size(2))
        x = self.relu1(x)
        x = self.fc1(x.float())
        x = self.relu2(x)
        x = self.dp(x)
        x = self.fc2(x)
        x = x / torch.norm(x)
        return x

class LSTMNet(nn.Module):

    def __init__(self):
        super(LSTMNet, self).__init__()
        self.LSTM_stack = nn.LSTM(num_dims, 64, num_layers=2, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.relu1 = nn.ReLU(True)
        self.fc1 = nn.Linear(512 * 64, 128)  ##  (max sentence length * hidden layer, 512)
        self.relu2 = nn.ReLU(True)
        self.dp = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 5)
        #self.bn = nn.BatchNorm1d()

    def forward(self, x):
        x, _ = self.LSTM_stack(x.float())  # (batch, sentence_len, hidden_units)
        #x2, _ = self.LSTM_stack(x2.float())
        #x = torch.cat((x1,x2),dim=1)
        #print(x.shape)
        #x = self.dp(x)
        #x = self.bn(x)
        

        # use every word in the sentence
        x = x.contiguous().view(-1, x.size(1) * x.size(2))
        #x = torch.cat((x,x3),dim=1)
        #print(x.shape,"shape")
        x = self.relu1(x)
        x = self.fc1(x.float())
        x = self.relu2(x)
        x = self.dp(x)
        x = self.fc2(x)
        x = x / torch.norm(x)
        return x

class FCNet(nn.Module):                 
    def __init__(self):
        super(FCNet, self).__init__()   
        self.fc1 = nn.Linear(3,32) 
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Linear(32,16)
        self.relu2 = nn.ReLU(True)
        self.fc3 = nn.Linear(16,5)
        
    def forward(self, input):
            output = input.view(-1, 3)
            output = self.fc1(output)
            output = self.relu1(output)
            output = self.fc2(output)
            output = self.relu2(output)
            output = self.fc3(output)
            
            return output