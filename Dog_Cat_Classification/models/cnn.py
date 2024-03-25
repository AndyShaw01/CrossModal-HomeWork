import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import ResNet50_Weights,ResNet34_Weights

import os
os.environ['TORCH_HOME'] = './models/model_pretrained'

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(2048, 2)
    
    def forward(self, x):
        return self.model(x)
    
    def __call__(self, x):
        return self.forward(x)
    
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(512, 2)
    
    def forward(self, x):
        return self.model(x)
    
    def __call__(self, x):
        return self.forward(x)
        
if __name__ == "__main__":
    model = ResNet50()
    print(model)
    # 构建一个随机的数据
    data = torch.randn(1, 3, 224, 224)
    output = model(data)
    print(output)
    print(output.size())