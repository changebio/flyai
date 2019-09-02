from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

class SimpleNet(nn.Module):

    """ Simple network"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4,32, kernel_size=3, padding=1), # 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 8 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 500),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64*8*6)
        x = self.classifier(x)
        return x

num_dims = 25*4*2    
class LSTMNet(nn.Module):

    def __init__(self):
        super(LSTMNet, self).__init__()
        self.LSTM_stack = nn.LSTM(num_dims, 128, num_layers=2, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.relu1 = nn.ReLU(True)
        self.fc1 = nn.Linear(32 * 128, 1024)  ##  (max sentence length * hidden layer, 512)
        self.relu2 = nn.ReLU(True)
        self.dp = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 500)
        #self.bn = nn.BatchNorm1d()

    def forward(self, x):
        x, _ = self.LSTM_stack(x)
        #x1, _ = self.LSTM_stack(x1.float())  # (batch, sentence_len, hidden_units)
        #x2, _ = self.LSTM_stack(x2.float())
        #x = torch.cat((x1,x2),dim=1)
        #x = self.dp(x)
        #x = self.bn(x)
        

        # use every word in the sentence
        x = x.contiguous().view(-1, x.size(1) * x.size(2))
        x = self.relu1(x)
        x = self.fc1(x.float())
        x = self.relu2(x)
        x = self.dp(x)
        x = self.fc2(x)
        x = x / torch.norm(x)
        return x
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.relu1 = nn.ReLU(True)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.relu2 = nn.ReLU(True)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)
        self.relu3 = nn.ReLU(True)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)
        output = self.bn1(output)
        output = self.pool1(output)

        output = self.conv2(output)
        output = self.relu2(output)
        output = self.bn2(output)
        output = self.pool2(output)

        output = output.view(-1, 64 * 4 * 4)
        output = self.fc1(output)
        output = self.relu3(output)
        output = self.fc2(output)

        return output
    
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1',
                        nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))

        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2',
                        nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class Net(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.2, final_drop_rate=0.2, num_classes=256):

        super(Net, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):

            # add dropout to the last dense block
            if i == len(block_config) - 1:
                dropout = drop_rate
            else:
                dropout = 0

            block = _DenseBlock(
                num_layers=num_layers, num_input_features=num_features,
                bn_size=bn_size, growth_rate=growth_rate, drop_rate=dropout
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2
                )
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.dropout = nn.Dropout(p=final_drop_rate)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=False)
        # I changed 7 -> 9 because 224 -> 299
        # no! the image size was changed back to 244 and then kernel_size changed to 9
        out = F.avg_pool2d(out, kernel_size=9).view(features.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out
