import torch
import torch.nn as nn

from torchvision.models import vit_b_16, ViT_B_16_Weights

from transformers import ViTForImageClassification, ViTFeatureExtractor
from transformers import TrainingArguments, Trainer

import os
os.environ['TORCH_HOME'] = './model_pretrained'

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.weights = ViT_B_16_Weights.DEFAULT
        self.model = vit_b_16(weights=self.weights)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, 2)
    
    def forward(self, x):
        return self.model(x)
    
    def __call__(self, x):
        return self.forward(x)

class ViT_transformers(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=2)

    def forward(self, x):
        return self.model(x)

    def __call__(self, x):
        return self.forward(x)
    
if __name__ == "__main__":
    model = ViT_transformers()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 构建一个随机的数据
    data = torch.randn(1, 3, 224, 224)
    output = model(data)
    print(output)
    # print(output.size())
