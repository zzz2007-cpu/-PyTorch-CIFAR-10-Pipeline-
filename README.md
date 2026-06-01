最终项目形态
你的仓库应该做成这样：

.
├── configs/
│   ├── simple_cnn_baseline.yaml
│   ├── simple_cnn_aug.yaml
│   ├── simple_cnn_cosine.yaml
│   ├── resnet18.yaml
│   └── label_noise_10.yaml
├── data/
│   └── cifar10.py
├── models/
│   ├── simple_cnn.py
│   └── resnet18.py
├── engine/
│   ├── trainer.py
│   ├── evaluator.py
│   └── checkpoint.py
├── analysis/
│   ├── plot_curves.py
│   ├── confusion_matrix.py
│   ├── badcase_analysis.py
│   └── label_noise_analysis.py
├── utils/
│   ├── config.py
│   ├── logger.py
│   ├── metrics.py
│   └── seed.py
├── outputs/
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
└── report.md
开发路线
第一阶段：先跑通最小训练闭环。

目标不是高准确率，而是完整跑通：

加载 CIFAR-10
-> DataLoader
-> SimpleCNN
-> forward
-> loss
-> backward
-> optimizer.step
-> validation
-> 保存 checkpoint
你需要实现：

python train.py --config configs/simple_cnn_baseline.yaml
python evaluate.py --checkpoint outputs/.../best.pt
这一阶段结束后，你应该能解释：

optimizer.zero_grad() 为什么需要
loss.backward() 做了什么
model.train() 和 model.eval() 的区别
验证阶段为什么要 torch.no_grad()
checkpoint 里保存了什么
第二阶段：把脚本升级成 Pipeline。

这一步是项目从“教程”变成“工程”的关键。

你要做到：

用 YAML 管理实验参数
支持 SimpleCNN / ResNet18 切换
支持 SGD / Adam 切换
支持 StepLR / CosineAnnealingLR
支持是否开启 data augmentation
支持 seed 固定
支持 checkpoint 保存 best / last
支持训练日志 CSV 或 TensorBoard
配置文件大概长这样：

seed: 42
dataset: cifar10
model: simple_cnn
epochs: 50
batch_size: 128
optimizer: sgd
lr: 0.1
weight_decay: 0.0005
scheduler: cosine
data_aug: true
num_workers: 4
第三阶段：做消融实验，这是简历核心。

至少做这些实验：

实验	目的
SimpleCNN baseline	建立基线
SimpleCNN + data augmentation	看数据增强是否提升泛化
SimpleCNN + weight decay	看正则化是否缓解过拟合
SGD vs Adam	对比优化器行为
No scheduler vs StepLR vs Cosine	分析学习率策略
SimpleCNN vs ResNet18	分析模型容量影响
batch size 64/128/256	看 batch size 对稳定性的影响
结果表不要只写准确率，要写成这样：

Model	Aug	Optimizer	Scheduler	Best Val Acc	Train Acc	Gap
SimpleCNN	No	SGD	None	xx%	xx%	xx%
SimpleCNN	Yes	SGD	Cosine	xx%	xx%	xx%
ResNet18	Yes	SGD	Cosine	xx%	xx%	xx%
这里的 Gap = Train Acc - Val Acc 很重要，它能体现你在分析过拟合，而不是只看最终分数。

第四阶段：做岗位贴合的“数据质量评估”。

这是你区别于普通 CIFAR-10 项目的地方。

加三个分析模块：

1. label noise 实验
   随机打乱 5%、10%、20% 训练标签，观察 loss、acc、gap 变化。

2. bad case analysis
   导出预测错误样本，记录 true label、pred label、confidence。

3. confusion matrix + per-class metrics
   看哪些类别容易混，例如 cat/dog、deer/horse、truck/automobile。
这部分最贴岗位里的：

训练数据和评估
Rubric
问题分解
评估标准
数据质量
模型表现分析
你甚至可以在 report.md 里写一个错误样本 Rubric：

错误类型	判断标准
类别相似	主体外形接近，如 cat/dog
图像模糊	主体边缘不清晰
背景干扰	背景占比过大
低分辨率歧义	32x32 下细节不足
疑似标注噪声	人眼也难以判断类别
第五阶段：写 README 和实验报告。

README 不要写成“怎么运行代码”就结束，要写成小型实验报告。

建议结构：

1. 项目简介
2. 岗位能力映射
3. 项目结构
4. 训练 Pipeline 设计
5. 实验配置
6. Ablation Study
7. Loss / Accuracy 曲线分析
8. Confusion Matrix 分析
9. Bad Case Analysis
10. Label Noise 数据质量实验
11. 总结与反思
report.md 里要写判断句，比如：

加入数据增强后，验证准确率提升，同时 train-val gap 下降，说明随机裁剪和水平翻转缓解了过拟合，提高了模型泛化能力。

在 10% label noise 设置下，训练 loss 下降速度变慢，验证准确率明显下降，说明标签质量会直接影响模型收敛稳定性和最终泛化表现。

混淆矩阵显示 cat/dog、deer/horse 是主要错误来源，说明模型在细粒度外观相似类