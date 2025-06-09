# CIFAR-10 图像分类 - 卷积神经网络实验

本项目旨在使用卷积神经网络（CNN）对 CIFAR-10 图像分类数据集进行建模和优化。通过对网络结构、训练策略和正则化手段的探索，构建了多个具有良好泛化能力的模型。


## 📌 项目结构与实验内容

### 1. 项目目标
构建一个高性能 CNN 模型，对 CIFAR-10 数据集进行图像分类。重点优化点包括：
- 网络结构设计（使用 Residual Block、BatchNorm、Dropout 等）
- 激活函数（ReLU、GeLU、SiLU 等）
- 损失函数与优化器选择（CrossEntropy、AdamW）
- 数据增强（随机裁剪、水平翻转、CutMix）
- 训练技巧（Lookahead、学习率调度器）

### 2. 模型对比与结果

| 模型 | 方法 | 测试准确率 |
|------|------|-------------|
| Model I | ResNet + BatchNorm + 数据增强 | 93.08% |
| Model II | Model I + CutMix + Lookahead + CosineLR | 94.45% |

### 3. 批归一化（BatchNorm）实验
对比基础 VGG 与 VGG_BatchNorm：
- **VGG_BN 收敛更快**，在验证集上准确率提高超过 5%
- **训练过程更稳定**，梯度方差和梯度差异显著降低，特别在使用 SGD 优化器时效果显著

## 🛠️ 技术栈

- PyTorch
- torchvision
- numpy / matplotlib 等
- Python 3.x

