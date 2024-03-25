# 猫狗分类系统

## 文件架构

```
.
├── Datasets
│   ├── train
│   └── val
├── Experiment          # 运行实验脚本
│   └── run.py  
├── Logs                # 实验log
│   ├── final           # 最终报告内的实验记录
│   └── test            # 测试记录
├── Scripts             # 启动脚本
│   ├── cnn_test.sh
│   ├── dnn_test.sh
│   └── vit_test.sh
├── model_pretrained    # 下载的预训练模型权重
│   └── hub
├── models              # 正式模型代码
│   ├── __pycache__
│   ├── cnn.py          # 预训练的ResNet34、ResNet50
│   ├── dnn.py          # 手动实现的DNN
│   └── vit.py          # 预训练的ViT-base
├── models_toy          # 手动实现的模型
│   ├── __init__.py
│   ├── __pycache__
│   ├── cnn.py          # 手动实现的ResNet34
│   └── vit.py          # 手动实现的ViT
└── readme.md
```

## 运行


```shell
nohup bash Scripts/your_test.py
```

## 数据集

- 2500张猫狗数据集