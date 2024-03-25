import os
import argparse
import sys
import logging

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(os.path.abspath('../Work_1/'))
from models_toy.cnn import ResNet
from models_toy.vit import ViT_Toy
from models.cnn import ResNet50, ResNet34
from models.dnn import DNN
# from models.vit import ViT_transformers
from models.vit import ViT

logging.basicConfig(
            level=logging.INFO, format='%(asctime)s %(message)s', datefmt='[%H:%M:%S]')

def main(args):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=args.train_folder_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=args.test_folder_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    if args.model == "cnn_toy_resnet34":
        model = ResNet()
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.model == "cnn_resnet50":
        model = ResNet50()
        model.to(device)
        optimizer = optim.SGD(model.model.fc.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.model == "cnn_resnet34":
        model = ResNet34()
        model.to(device)
        optimizer = optim.SGD(model.model.fc.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.model == "dnn":
        model = DNN()
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.model == "ViT":
        model = ViT()
        model.to(device)
        optimizer = optim.SGD(model.model.heads.head.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.model == "ViT_toy":
        model = ViT_Toy(image_size=224, patch_size=16, patch_dim=768, dim=768, heads=12, mlp_dim=3072, depth=6, num_classes=2)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        
    # train
    logging.info('开始训练')
    for epoch in range(args.epochs): 
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            logging.info(f'Epoch {epoch+1}, Loss: {loss.item()}')
    logging.info('训练完成')
    # test
    correct = 0
    total = 0

    model.eval()   
    logging.info('开始测试')     
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    logging.info(f'Accuracy: {100 * correct / total}%')
    logging.info('测试完成')
    # torch.save(model.state_dict(), './model_trained/dnn_v1.pth')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=32, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--learning-rate", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 10)")
    
    parser.add_argument("--model", type=str, default="cnn", metavar="N", help="model to use")
    
    parser.add_argument("--train-folder-path", type=str, default="./Datasets/train", metavar="N", help="train folder path")
    parser.add_argument("--test-folder-path", type=str, default="./Datasets/val", metavar="N", help="train folder path")
    parser.add_argument("--model-saved-path", type=str, default="./model_trained", help="path to save model")

    args = parser.parse_args()

    main(args)