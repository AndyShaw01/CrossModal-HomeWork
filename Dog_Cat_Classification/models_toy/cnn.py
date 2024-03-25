import torch 
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable as V

class ResidualBlock_toy(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut
    
    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
    
class ResNet(nn.Module):
    def __init__(self, num_class=2):
        super(ResNet, self).__init__()
        # img transform
        self.pre_transform = nn.Sequential(
            nn.Conv2d(3, 64, 7,2,3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # residual block
        self.layer1 = self._make_layer(64, 256, 3)
        self.layer2 = self._make_layer(256, 1024, 4, stride = 2)
        self.layer3 = self._make_layer(1024, 4096, 6, stride = 2)
        self.layer4 = self._make_layer(4096, 1024, 3, stride = 2)

        # full connect
        self.fc = nn.Linear(1024, num_class)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # make layer with multi residual blocl
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock_toy(inchannel, outchannel, stride, shortcut))

        for _ in range(1, block_num):
            layers.append(ResidualBlock_toy(outchannel, outchannel))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.pre_transform(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    

if __name__ == "__main__":
    model = ResNet()
    input = V(torch.randn(1,3,224,224))
    out = model(input)
    print(out.size()) # 输出：torch.Size([1, 1000])