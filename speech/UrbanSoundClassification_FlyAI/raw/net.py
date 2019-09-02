# -*- coding: utf-8 -*
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, stride=(1, 1)),
            nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
            nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5),
                                   nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=48, kernel_size=5),
                                   nn.ReLU())
        self.linear1 = nn.Linear(2400, 64)
        self.linear2 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.softmax(self.linear2(x))

        return x
