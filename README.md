# CIFAR-10 图像分类训练 Pipeline 与实验分析

这是一个基于 PyTorch 的 CIFAR-10 图像分类训练与评估项目。项目不仅实现模型训练，还围绕实验配置、checkpoint 管理、消融实验、曲线可视化、混淆矩阵和错误样本分析构建了完整 Pipeline。

## 功能概览

- YAML 驱动实验配置，支持模型、优化器、学习率策略、数据增强和标签噪声切换。
- 支持 SimpleCNN 与 ResNet18，训练过程保存 `best.pt` 和 `last.pt`。
- 自动记录 `metrics.csv`，可生成 loss/accuracy 曲线。
- 完成 SGD vs Adam、None vs StepLR vs CosineLR、数据增强、模型容量等消融实验。
- 输出 confusion matrix、per-class accuracy 和 badcase 样本，用于分析模型错误模式。
- 预留 label noise 实验，用于评估训练数据质量下降对模型泛化的影响。

## 项目结构

```text
.
├── configs/                 # YAML 实验配置
├── data/cifar10.py          # 数据加载、数据增强、标签噪声
├── models/                  # SimpleCNN / ResNet18
├── engine/                  # 训练、验证、checkpoint
├── analysis/                # 曲线、混淆矩阵、错误样本分析
├── outputs/                 # 实验输出
├── train.py                 # 训练入口
├── evaluate.py              # checkpoint 评估入口
├── experiments_summary.csv  # 实验汇总表
└── report.md                # 实验报告
```

## 快速运行

训练最优 SimpleCNN 配置：

```bash
python train.py --config configs/simple_cnn_cosine.yaml
```

评估 checkpoint：

```bash
python evaluate.py --checkpoint outputs/simple_cnn_cosine_20260601_125946/best.pt
```

从中断处恢复训练：

```bash
python train.py --config configs/resnet18.yaml --resume outputs/resnet18_cosine_20260601_172941/last.pt
```

运行 10% 标签噪声实验：

```bash
python train.py --config configs/simple_cnn_label_noise_10.yaml
```

## 实验结果

完整表格见 [experiments_summary.csv](experiments_summary.csv)，详细分析见 [report.md](report.md)。

| 实验 | Best Val Acc | Best Epoch | Last Train Acc | Last Val Acc | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| SimpleCNN Baseline | 79.33% | 18 | 88.27% | 76.60% | 11.67% |
| SimpleCNN + Aug | 78.43% | 19 | 78.47% | 76.70% | 1.77% |
| SimpleCNN + Adam | 79.48% | 17 | 92.26% | 78.90% | 13.36% |
| SimpleCNN + StepLR | 81.80% | 12 | 95.43% | 81.53% | 13.90% |
| SimpleCNN + CosineLR | 86.79% | 50 | 89.38% | 86.79% | 2.59% |
| ResNet18 + CosineLR | 84.41% | 19 | 87.60% | 77.53% | 10.07% |

最优实验是 `SimpleCNN + CosineLR`，验证准确率达到 86.79%。相比 baseline，它将最终 train-val gap 从 11.67% 降低到 2.59%，说明数据增强和 cosine 学习率退火有效提升了泛化能力。

## 曲线图

SimpleCNN + CosineLR：

![SimpleCNN Cosine Accuracy](outputs/simple_cnn_cosine_20260601_125946/accuracy_curve.png)

![SimpleCNN Cosine Loss](outputs/simple_cnn_cosine_20260601_125946/loss_curve.png)

ResNet18 + CosineLR：

![ResNet18 Accuracy](outputs/resnet18_cosine_20260601_172941/accuracy_curve.png)

![ResNet18 Loss](outputs/resnet18_cosine_20260601_172941/loss_curve.png)

## 混淆矩阵与错误分析

最优模型的混淆矩阵：

![Confusion Matrix](outputs/simple_cnn_cosine_20260601_125946/confusion_matrix.png)

Per-class accuracy 显示，`automobile`、`truck`、`ship` 等类别识别较稳定，而 `cat`、`bird`、`dog` 等类别准确率较低。主要错误对包括：

| True | Pred | Count |
| --- | --- | ---: |
| automobile | truck | 13 |
| dog | cat | 13 |
| ship | airplane | 12 |
| frog | cat | 9 |
| truck | automobile | 9 |
| deer | horse | 7 |

错误样本导出到：

```text
outputs/simple_cnn_cosine_20260601_125946/badcases/
```

这些 bad cases 表明，模型错误主要来自低分辨率下的类别外观相似、主体边界模糊、背景干扰和高置信错误。

## 分析脚本

生成训练曲线：

```bash
python analysis/plot_curves.py --metrics outputs/simple_cnn_cosine_20260601_125946/metrics.csv
```

生成混淆矩阵和 per-class accuracy：

```bash
python analysis/confusion_matrix.py --checkpoint outputs/simple_cnn_cosine_20260601_125946/best.pt --num-workers 0
```

导出错误样本：

```bash
python analysis/badcase_analysis.py --checkpoint outputs/simple_cnn_cosine_20260601_125946/best.pt --num-workers 0
```

## 结论

实验结果说明，单纯更换优化器并不一定带来稳定泛化提升；数据增强能显著降低过拟合，而 CosineAnnealingLR 能进一步提升验证准确率。ResNet18 在当前 20 epoch 配置下峰值较高但波动明显，说明更大模型需要更长训练和更细致的超参数控制。

本项目围绕 CIFAR-10 图像分类任务构建了完整实验流程，覆盖配置管理、训练恢复、实验对比、曲线可视化、混淆矩阵分析、错误样本导出和标签噪声实验，便于复现实验结果并分析不同训练策略的影响。
