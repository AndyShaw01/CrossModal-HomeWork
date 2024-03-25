# 这里将使用Pytorch构建一个dnn的模型，来做猫狗分类
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(3*224*224, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        # x = self.dropout(x)
        x = F.relu(self.fc4(x))

        return x
    
if __name__ == "__main__":
    model = DNN()
    print(model)
    # 构建一个随机的数据
    data = torch.randn(1, 3, 53, 53)
    output = model(data)
    print(output)