# CIFAR-10 图像分类训练 Pipeline 实验报告

## 1. 项目概述

本项目基于 PyTorch 实现 CIFAR-10 图像分类训练 Pipeline，并围绕模型训练、实验配置、结果追踪和错误分析形成完整实验闭环。项目重点不是单次训练准确率，而是展示机器学习工程中更真实的工作方式：配置化实验、可复现训练、指标对比、可解释评估和数据质量分析。

本项目最终完成了：

- SimpleCNN 与 ResNet18 两种模型训练。
- SGD / Adam 优化器对比。
- None / StepLR / CosineAnnealingLR 学习率策略对比。
- 数据增强对泛化能力的影响分析。
- 10% label noise 数据质量实验。
- confusion matrix、per-class accuracy、bad case analysis。
- checkpoint 保存、断点恢复、CSV 日志记录和曲线可视化。

## 2. 工程结构

```text
configs/                 YAML 实验配置
data/cifar10.py          CIFAR-10 数据集、数据增强、label noise 注入
models/                  SimpleCNN 与 ResNet18
engine/trainer.py        单轮训练与验证逻辑
engine/checkpoint.py     checkpoint 保存与加载
analysis/                曲线绘制、混淆矩阵、错误样本导出
results/                 汇总后的实验结果
docs/assets/             README 和报告展示用图表
outputs/                 本地原始实验输出
```

训练命令示例：

```bash
python train.py --config configs/simple_cnn_cosine.yaml
```

断点恢复命令示例：

```bash
python train.py --config configs/resnet18.yaml --resume outputs/resnet18_cosine_20260601_172941/last.pt
```

## 3. 实验设计

本项目围绕五个问题设计实验：

| 问题 | 对应实验 |
| --- | --- |
| 优化器是否影响收敛和泛化？ | SGD vs Adam |
| 学习率策略是否提升验证集表现？ | None vs StepLR vs CosineAnnealingLR |
| 数据增强是否缓解过拟合？ | no augmentation vs augmentation |
| 模型容量是否带来更好结果？ | SimpleCNN vs ResNet18 |
| 标签质量下降会带来多大影响？ | clean labels vs 10% label noise |

## 4. 实验结果总表

`Gap = Last Train Acc - Last Val Acc`。Gap 越大，通常说明模型越可能过拟合训练集。

| Experiment | Best Val Acc | Best Epoch | Last Train Acc | Last Val Acc | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| SimpleCNN Baseline | 79.33% | 18 | 88.27% | 76.60% | 11.67% |
| SimpleCNN + Aug | 78.43% | 19 | 78.47% | 76.70% | 1.77% |
| SimpleCNN + Adam | 79.48% | 17 | 92.26% | 78.90% | 13.36% |
| SimpleCNN + StepLR | 81.80% | 12 | 95.43% | 81.53% | 13.90% |
| **SimpleCNN + CosineLR** | **86.79%** | 50 | 89.38% | 86.79% | **2.59%** |
| ResNet18 + CosineLR | 84.41% | 19 | 87.60% | 77.53% | 10.07% |
| SimpleCNN + 10% Label Noise | 82.45% | 19 | 74.27% | 82.38% | -8.11% |

完整 CSV 见 [results/experiments_summary.csv](results/experiments_summary.csv)。

## 5. 消融实验分析

### 5.1 优化器：SGD vs Adam

Adam 在 20 epoch 内收敛较快，Best Val Acc 为 **79.48%**，略高于 baseline 的 **79.33%**。但 Adam 最终训练准确率达到 **92.26%**，验证准确率为 **78.90%**，Gap 为 **13.36%**，说明它在训练集上拟合更强，但泛化提升有限。

SGD 在加入学习率策略后表现更稳。`SimpleCNN + StepLR` 达到 **81.80%**，`SimpleCNN + CosineLR` 达到 **86.79%**。这说明在当前任务中，学习率调度带来的收益比单纯替换 Adam 更明显。

### 5.2 Scheduler：None vs StepLR vs Cosine

无 scheduler 的 baseline 最佳验证准确率为 **79.33%**。StepLR 提升到 **81.80%**，说明阶段性降低学习率能改善中后期验证表现。

CosineAnnealingLR 配合数据增强达到 **86.79%**，并且最终 Gap 只有 **2.59%**。这说明平滑退火让模型在训练后期继续获得泛化收益，而不是只提高训练集拟合程度。

### 5.3 数据增强是否提升泛化

只加入数据增强但不加 scheduler 时，Best Val Acc 为 **78.43%**，没有超过 baseline。但它将最终 Gap 从 **11.67%** 降低到 **1.77%**，说明随机裁剪和水平翻转有效抑制了过拟合。

当数据增强与 CosineAnnealingLR 结合时，验证准确率提升到 **86.79%**。因此，数据增强的价值主要体现在提升泛化稳定性；要转化成更高验证准确率，还需要合适的训练策略配合。

### 5.4 SimpleCNN vs ResNet18

ResNet18 在 20 epoch 内最高达到 **84.41%**，低于 SimpleCNN + CosineLR 的 **86.79%**。同时，ResNet18 第 19 轮达到峰值，第 20 轮验证准确率回落到 **77.53%**，说明更大容量模型在当前训练时长和超参数下更不稳定。

这个现象不代表 ResNet18 本身弱，而是说明大模型对学习率、训练轮数和正则化更敏感。对于简历项目而言，这个结论比单纯追求更大模型更有价值。

## 6. 曲线分析

### 6.1 最优模型：SimpleCNN + CosineLR

![SimpleCNN Cosine Accuracy](docs/assets/simplecnn_cosine_accuracy.png)

![SimpleCNN Cosine Loss](docs/assets/simplecnn_cosine_loss.png)

该模型验证准确率随训练逐步提升，最终达到 **86.79%**。最终 train-val gap 仅 **2.59%**，说明模型没有明显过拟合，是当前配置下最稳的结果。

### 6.2 ResNet18 曲线

![ResNet18 Accuracy](docs/assets/resnet18_accuracy.png)

![ResNet18 Loss](docs/assets/resnet18_loss.png)

ResNet18 曲线后期验证准确率存在明显波动，说明模型容量增加后，训练稳定性成为更关键的问题。

### 6.3 Label Noise 曲线

![Label Noise Accuracy](docs/assets/label_noise_10_accuracy.png)

![Label Noise Loss](docs/assets/label_noise_10_loss.png)

10% label noise 实验中，训练集有 5000 / 50000 个标签被随机替换为错误类别。该实验最佳验证准确率为 **82.45%**，相比干净数据的 **86.79%** 下降 **4.34 个百分点**。

同时，最终训练准确率只有 **74.27%**，低于验证准确率 **82.38%**。这是合理现象：训练集标签被污染后，模型无法同时拟合全部错误标签；验证集仍然保持干净，因此验证准确率可能高于 noisy train accuracy。这个结果说明标签质量会直接影响模型收敛和最终泛化上限。

## 7. Confusion Matrix 与类别表现

最优模型的混淆矩阵如下：

![Confusion Matrix](docs/assets/simplecnn_cosine_confusion_matrix.png)

Per-class accuracy 显示，交通工具类别更容易识别：

| Class | Accuracy |
| --- | ---: |
| automobile | 93.60% |
| truck | 92.00% |
| ship | 91.90% |
| frog | 90.80% |

较困难的类别主要集中在动物类：

| Class | Accuracy |
| --- | ---: |
| cat | 73.50% |
| bird | 80.10% |
| dog | 81.60% |
| deer | 86.70% |

这说明模型对轮廓明显、背景模式较稳定的类别识别更好，而对细粒度外观相似、姿态变化大的动物类别更容易出错。

## 8. Bad Case Analysis

导出的错误样本显示，主要混淆对如下：

| True | Pred | Count | 可能原因 |
| --- | --- | ---: | --- |
| automobile | truck | 13 | 车辆轮廓相似，低分辨率下细节不足 |
| dog | cat | 13 | 动物局部纹理和姿态相似 |
| ship | airplane | 12 | 天空/海面背景和长条形主体带来干扰 |
| frog | cat | 9 | 背景和主体颜色干扰 |
| truck | automobile | 9 | 同属车辆类别，局部结构相似 |
| deer | horse | 7 | 四足动物轮廓相似 |

示例 bad cases：

| automobile → truck | dog → cat | ship → airplane | deer → horse |
| --- | --- | --- | --- |
| ![](docs/assets/badcases/automobile_to_truck.png) | ![](docs/assets/badcases/dog_to_cat.png) | ![](docs/assets/badcases/ship_to_airplane.png) | ![](docs/assets/badcases/deer_to_horse.png) |

错误样本可以归纳为四类：

| 错误类型 | 说明 |
| --- | --- |
| 类别外观相似 | cat/dog、deer/horse 等细粒度差异小 |
| 低分辨率歧义 | CIFAR-10 图像只有 32x32，关键细节不足 |
| 背景干扰 | 背景纹理或颜色占比大，影响主体判断 |
| 高置信错误 | 模型对错误类别非常确定，说明学到的特征存在系统性偏差 |

## 9. 数据质量实验：10% Label Noise

标签噪声实验通过 `data/cifar10.py` 中的 `apply_label_noise()` 实现，只污染训练集标签，验证集保持干净。配置如下：

```yaml
data:
  data_aug: true
  label_noise_rate: 0.10
  label_noise_seed: 42
```

对比结果：

| Setting | Best Val Acc | Last Train Acc | Last Val Acc |
| --- | ---: | ---: | ---: |
| Clean labels | 86.79% | 89.38% | 86.79% |
| 10% noisy labels | 82.45% | 74.27% | 82.38% |

结论：10% 训练标签噪声使最佳验证准确率下降 **4.34 个百分点**，说明标签质量对模型泛化有直接影响。该实验让项目从普通分类任务扩展到了训练数据质量评估，更贴近真实机器学习工程中的数据问题。

## 10. 总结

本项目完成了一个可复现、可分析、可扩展的 CIFAR-10 训练 Pipeline。最终结论如下：

- **最佳配置**：SimpleCNN + Data Augmentation + CosineAnnealingLR，Best Val Acc **86.79%**。
- **泛化分析**：数据增强显著降低 train-val gap，CosineLR 进一步提升验证准确率。
- **优化器结论**：Adam 收敛快，但当前实验下更容易出现较大泛化 gap。
- **模型容量结论**：ResNet18 峰值较高但波动明显，大模型需要更细致的训练策略。
- **错误分析结论**：主要错误来自动物类别混淆、车辆类别混淆、低分辨率和背景干扰。
- **数据质量结论**：10% label noise 使最佳验证准确率下降 4.34 个百分点，说明噪声标签会显著压低模型泛化上限。

这个项目最终展示的是完整机器学习实验能力：不只是训练模型，而是能设计实验、复现实验、解释结果、定位错误，并从数据质量角度分析模型表现。
