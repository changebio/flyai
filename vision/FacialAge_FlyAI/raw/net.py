## build CNN
from torch import nn
import torch
import torchvision
## build CNN
# =============================================================================
# class Net(nn.Module):                 
#     #def __init__(self,num_classes=10):
#     def __init__(self):
#         super(Net, self).__init__()   
#         self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)       
#         self.relu1=nn.ReLU(True)
#         self.bn1=nn.BatchNorm2d(32) 
#         self.pool1 = nn.MaxPool2d(2, 2)        
#         self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
#         self.relu2=nn.ReLU(True)
#         self.bn2=nn.BatchNorm2d(64) 
#         self.pool2 = nn.MaxPool2d(2, 2)   
#         self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
#         self.relu3=nn.ReLU(True)
#         self.bn3=nn.BatchNorm2d(128) 
#         self.pool3 = nn.MaxPool2d(2, 2)    
#         self.fc1 = nn.Linear(128*16*16, 512) 
#         self.relu4=nn.ReLU(True)
#         self.fc2 = nn.Linear(512,99)
# 
#     def forward(self, input):
#         output = self.conv1(input)
#         output = self.relu1(output)
#         output = self.bn1(output)
#         output = self.pool1(output)
#         
#         output = self.conv2(output)
#         output = self.relu2(output)
#         output = self.bn2(output)
#         output = self.pool2(output)
# 
#         output = self.conv3(output)
#         output = self.relu3(output)
#         output = self.bn3(output)
#         output = self.pool3(output)
#         
#         output = output.view(-1, 128*16*16)
#         output = self.fc1(output)
#         output = self.relu4(output)
#         output = self.fc2(output)
#         
#         return output
# =============================================================================

class Net(torch.nn.Module):
    """Bilinear CNN model.
    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (num_classes).
    The network accepts a 3*448*448 input, and the pool5 activation has shape
    512*28*28 since we down-sample 5 times.
    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: self.num_classes.
    """
    def __init__(self, num_classes=99, pretrained=True):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        self.num_classes = num_classes
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=pretrained).features
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(512**2, self.num_classes)

        # Freeze all previous layers.
        if pretrained:
            for param in self.features.parameters():
                param.requires_grad = False
            def init_weights(layer):
                if type(layer) == torch.nn.Conv2d or type(layer) == torch.nn.Linear:
                    torch.nn.init.kaiming_normal_(layer.weight.data)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias.data, val=0)
            self.fc.apply(init_weights)
            self.trainable_params = [
                {'params': self.fc.parameters()}
            ]
        else:
            self.trainable_params = self.parameters()

    def forward(self, X):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape N*3*448*448.
        Returns:
            Score, torch.autograd.Variable of shape N*self.num_classes.
        """
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        assert X.size() == (N, 512, 28, 28)
        X = X.view(N, 512, 28**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28**2)  # Bilinear
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, self.num_classes)
        return X

