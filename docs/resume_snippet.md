# Resume Snippet

## 中文简历版

**基于 PyTorch 的 CIFAR-10 图像分类训练 Pipeline 与实验分析**

- 构建可复现实验训练 Pipeline，支持 YAML 配置、SimpleCNN/ResNet18 切换、SGD/Adam、StepLR/CosineAnnealingLR、数据增强、标签噪声实验、checkpoint 保存与断点恢复。
- 完成多组消融实验，最佳模型 `SimpleCNN + Data Augmentation + CosineLR` 在 CIFAR-10 上达到 **86.79%** 验证准确率，并将最终 train-val gap 从 baseline 的 **11.67%** 降低到 **2.59%**。
- 实现 confusion matrix、per-class accuracy 与 bad case 导出，定位 `cat/dog`、`automobile/truck`、`ship/airplane` 等主要混淆模式。
- 设计 10% label noise 数据质量实验，验证噪声标签使最佳验证准确率从 **86.79%** 降至 **82.45%**，量化训练数据质量对模型泛化的影响。

## English Resume Version

**PyTorch CIFAR-10 Training Pipeline and Experimental Analysis**

- Built a reproducible image classification pipeline with YAML configs, model/optimizer/scheduler switching, data augmentation, label-noise experiments, checkpointing, and resume training.
- Conducted ablation studies across SGD/Adam, StepLR/CosineLR, data augmentation, SimpleCNN, and ResNet18; achieved **86.79%** best validation accuracy with SimpleCNN + augmentation + CosineLR.
- Added confusion matrix, per-class metrics, and bad-case export to analyze failure patterns such as cat/dog, automobile/truck, and ship/airplane confusion.
- Evaluated training-data quality with 10% label noise, where best validation accuracy dropped from **86.79%** to **82.45%**, demonstrating the impact of noisy labels on generalization.
