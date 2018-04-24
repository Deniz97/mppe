import torch
import torch.nn as nn
import torchvision

from torch.autograd import Variable

class FPN(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(FPN, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)

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

def build():
    model = FPN()
    model.cuda()
    return model

if __name__ == '__main__':
    model = build()
    x = Variable(torch.randn(1,3,480,480)).cuda()
    y = model.forward(x)

    print [y_.shape for y_ in y]