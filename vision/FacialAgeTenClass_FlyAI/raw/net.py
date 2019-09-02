## build CNN
from torch import nn
import torch
import torchvision
## build CNN

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
    def __init__(self, num_classes=10, pretrained=True):
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

