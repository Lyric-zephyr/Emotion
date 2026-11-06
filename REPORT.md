# 动物情绪识别系统：技术分析报告

**版本**：V5.1

**任务**：猫/狗图像的 Angry、Sad、happy 三分类

**方法**：ImageNet 迁移学习 + 三种 CNN 架构对比 + 软投票集成

**文档复核日期**：2026-09-29

## 1. 项目概述

系统将宠物面部图片映射到三个情绪类别。数据采用 PyTorch `ImageFolder` 目录结构，分别从 `Master-Folder/train`、`Master-Folder/val` 和 `Master-Folder/test` 读取。现有训练日志记录的数据规模为训练集 2100 张、验证集 450 张、测试集 450 张，每类在验证集和测试集中各 150 张。验证集用于选择最佳 checkpoint，测试集不参与梯度更新。

项目支持：

- DenseNet-121、ResNet-50、EfficientNet-B3 三种 ImageNet 预训练模型；
- 单模型训练、三模型对比训练和 TTA（Test-Time Augmentation）；
- 单模型推理、按验证集精度加权的集成推理；
- CLI 和 Gradio Web 两种推理入口。

## 2. 模型设计

| 模型 | 约参数量 | 分类头改造 | 设计特点 |
|:---|---:|:---|:---|
| DenseNet-121 | 7.0M | Dropout(0.3) + Linear | 密集连接、特征复用 |
| ResNet-50 | 23.5M | Dropout(0.3) + Linear | 残差连接、深层表达 |
| EfficientNet-B3 | 10.7M | Dropout(0.3) + Linear | 复合缩放、精度/效率平衡 |

除分类头外，网络使用 ImageNet 预训练参数并进行全参数微调。checkpoint 记录模型类型、输入尺寸、类别名、最佳验证精度、对应测试精度和 epoch，因此推理模块可以自动恢复架构，不依赖额外配置文件。

## 3. 训练策略

当前实现（`train.py`）的主要配置如下：

| 项目 | 配置 |
|:---|:---|
| 输入 | 384×384 RGB |
| 优化器 | AdamW，lr=3e-4，weight decay=0.01 |
| 学习率 | 3 epoch 线性 warmup + CosineAnnealingLR |
| 损失 | CrossEntropyLoss，label smoothing=0.08 |
| 正则化 | 分类头 Dropout=0.3，EMA decay=0.999 |
| 数据增强 | Flip、Rotation、ColorJitter、RandAugment、RandomErasing |
| 批级增强 | MixUp（α=0.15，概率=0.4） |
| 加速 | CUDA AMP；无 CUDA 时自动回退 CPU |
| 默认轮数/批大小 | 30 / 16 |

训练每轮记录 Train Loss、Train Acc、Val Acc 和 EMA Test Acc。当验证集精度提升时保存 checkpoint。checkpoint 中保存的是当轮模型参数，同时记录最佳验证精度和该轮 EMA 测试精度。`--model all` 按 EfficientNet-B3、ResNet-50、DenseNet-121 顺序完成三模型训练，并额外生成对比曲线和集成评估。

## 4. 评估与集成

单模型评估包含：

1. 每轮验证集和测试集准确率；
2. 训练结束时内存中模型的详细分类报告；
3. 训练结束时模型的测试集混淆矩阵；
4. 可选的 TTA 测试准确率。

集成推理对每个模型的 softmax 概率进行加权平均：

$$
P(y\mid x)=\frac{\sum_k w_k P_k(y\mid x)}{\sum_k w_k}
$$

其中 `w_k` 设计为模型的最佳验证集准确率。现有日志对应的权重是 ResNet-50 95.33、EfficientNet-B3 95.78、DenseNet-121 95.56；命令行的 `--equal` 可切换为等权平均。不同架构的特征提取和错误模式具有互补性，因此集成结果可作为默认 Web 推理方式。

## 5. 结果与产物

### 5.1 历史训练结果

下表来自 `results/logs/training.log` 中 2026-07-20、seed=42、30 epochs、384×384 输入的三模型训练记录：

| 模型 | Best Val | 最佳 Val 轮次 | 该轮 Test（EMA） | 最终 Test（EMA） | 训练耗时 |
|:---|---:|---:|---:|---:|---:|
| EfficientNet-B3 | 95.78% | 13 | 93.33% | 95.11% | 12.6 min |
| ResNet-50 | 95.33% | 29 | 94.89% | 95.11% | 9.9 min |
| DenseNet-121 | 95.56% | 25 | 95.11% | 94.67% | 11.0 min |

训练耗时仅代表该次 CUDA 环境，不能作为跨设备性能基准。

### 5.2 集成复测结果（既有记录）

本节沿用既有项目文档中的 2026-09-29 复测记录，仓库未附该次复测的独立日志。记录的评估条件为：使用 `results/models/` 中的现有 checkpoint，在 `Master-Folder/test` 的 450 张图片（每类 150 张）上运行与推理代码相同的预处理和 softmax 融合；环境为 PyTorch 2.14.0+cu130、NVIDIA RTX 5060。结果如下：

| 方法 | 正确数/总数 | Test Acc |
|:---|---:|---:|
| ResNet-50 | 428/450 | 95.1111% |
| EfficientNet-B3 | 427/450 | 94.8889% |
| DenseNet-121 | 425/450 | 94.4444% |
| 等权软投票 | 432/450 | **96.0000%** |
| 验证集精度加权软投票 | 432/450 | **96.0000%** |

相对最佳单模型，集成提升 `0.8889` 个百分点，即多正确识别 4 张测试图片。这个提升是固定数据划分和固定 checkpoint 上的一次实测，不应外推成普遍保证。结果来自 2026-09-29 对仓库现有权重的完整测试集复测。

### 5.3 结果文件

仓库中随项目提供的历史训练结果包括三模型 checkpoint、训练日志、三张混淆矩阵和模型对比图。它们现在按用途存放在：

```text
results/
├── models/
│   ├── best_model.pth
│   ├── best_model_densenet121.pth
│   ├── best_model_efficientnet_b3.pth
│   └── best_model_resnet50.pth
├── logs/
│   └── training.log
└── plots/
    ├── confusion_densenet121.png
    ├── confusion_efficientnet_b3.png
    ├── confusion_resnet50.png
    └── model_comparison_test_acc.png
```

后续运行训练脚本会继续写入上述目录；代码不会再把训练结果写到项目根目录。单模型训练还会在 `results/plots/` 生成 `test_acc_<model>.png`。`best_model.pth` 是 EfficientNet-B3 checkpoint 的便捷副本。

现有四个 checkpoint 的内部 `version` 字段仍为 `V5.0`，因为它们由旧版保存逻辑生成；当前源代码已统一为 V5.1，新训练产生的 checkpoint 将写入 `V5.1`。版本元数据差异不影响模型结构识别或权重加载。

## 6. 复现实验

```bash
./scripts/setup_venv.sh
./scripts/train.sh --model all --epochs 30 --seed 42
```

只训练单个模型时，例如：

```bash
./scripts/train.sh --model resnet50 --epochs 30 --batch-size 16
```

训练完成后可以直接使用：

```bash
./scripts/predict.sh --ensemble path/to/image.jpg
./scripts/app.sh
```

## 7. 工程约定与限制

- 依赖版本以 `requirements.txt` 为准，建议始终在项目 `.venv` 中运行。
- 首次构建模型可能需要下载 torchvision 的 ImageNet 权重；离线环境可预先缓存权重。
- 结果目录中的 `.pth` 文件体积较大，部署时可只保留需要的模型；集成推理至少需要两个模型。
- 训练代码每个 epoch 都评估测试集。测试指标不参与反向传播或 checkpoint 选择，但正式实验中仍不应据此调参，否则会引入测试集信息泄漏。
- checkpoint 的 `test_accuracy` 表示最佳验证轮次对应的 EMA 测试值，不是最后一轮测试值，也不是重新加载最佳 checkpoint 后独立评估的结果。
- 报告中的准确率应以对应运行的 `results/logs/training.log` 为准，避免将不同数据划分或超参数实验混用。
