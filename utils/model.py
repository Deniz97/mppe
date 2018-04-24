import torch
import torch.nn as nn
import torchvision

from torch.autograd import Variable

# We can handle FPN and post-steps in different classes for further usability of FPN.
# FPN paper: https://arxiv.org/abs/1612.03144
# A new idea: https://arxiv.org/pdf/1803.01534.pdf

class FPN(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FPN, self).__init__()
        self.feature_size = 256
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        c2 = self.resnet.layer1(x)
        c3 = self.resnet.layer2(c2)
        c4 = self.resnet.layer3(c3)
        c5 = self.resnet.layer4(c4)

        # feature pyramid network computations here

        return c2, c3, c4, c5

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.FPN = FPN()


    def forward(self, x):
        features = FPN(x)
        p2, p3, p4, p5 = features

        return features

def build():
    model = Model()
    model.cuda()
    return model

if __name__ == '__main__':
    model = build()
    x = Variable(torch.randn(1,3,480,480)).cuda()
    y = model.forward(x)

    print [y_.shape for y_ in y]