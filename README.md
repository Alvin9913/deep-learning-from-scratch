# Deep Learning From Scratch

六个实验，全部不调 `torchvision.models`、不调框架的卷积层和梯度——网络结构一层层搭，反向传播一行行写。每个实验都有训练曲线和实测指标，报告在各自目录里。

## 结果一览

| 实验 | 模型 | 关键结果 |
|---|---|---|
| q1 | 可配置 MLP（手写前向/反向） | Acc **0.9686** / F1 0.9683，超过 Random Forest 基线（0.9503） |
| q2 | VGG-16 从零实现 | Acc 0.5847 / F1 0.5801（3 epoch，欠拟合明显） |
| q3 | ResNet18 从零，残差 vs 普通对照 | BasicBlock **0.7738** vs PlainBlock 0.7197 |
| q4 | numpy 手写 MLP | 测试集 **93.33%** |
| q5 | numpy 手写卷积（im2col） | 测试集 **97.77%**，5 epoch / 每 epoch 22.5s |
| q6 | 自写 tokenizer + Dataset，IMDb 情感分类 | 测试 **0.8480** / F1 0.8503 |

## q1 · 可配置 MLP

`q1-mlp/mlp-from-scratch.ipynb`

- 手工实现前向传播、交叉熵、反向传播与 SGD 更新，没有用 `nn.Linear`。
- 做了三组控制变量实验：
  - 隐藏层宽度：`[64,32]` 0.9576 → `[256,128]` 0.9651 → `[512,256]` 0.9692
  - 隐藏层深度：1 层 0.9656 → 2 层 0.9676 → 3 层 0.9686
  - 与 Random Forest 对比：0.9686 vs 0.9503
- 结论：在这个规模的数据上，宽度带来的收益比深度更大，而且 2 层之后基本饱和。

## q2 · VGG-16 从零实现

`q2-vgg16/vgg16-from-scratch.ipynb`

- 按 Simonyan & Zisserman (2014) 的 Configuration D，把 13 个卷积层 + 3 个全连接层全部写出来。
- 同时训练一个 SimpleCNN 做对照，看层数增加到底带来了什么。
- 实测 Acc 0.5847 / F1 0.5801：**这个结果是欠拟合的**，训练曲线还没收敛就停了。报告里分析了原因，也写明了要跑满需要多少资源——这一点我没有粉饰。

## q3 · ResNet18 与残差消融

`q3-resnet18/resnet18-from-scratch.ipynb`

- 从零实现 ResNet18，同时实现两种 block：`BasicBlock`（带 skip connection）和 `PlainBlock`（不带）。
- 相同训练配置下对比：**0.7738 vs 0.7197，残差带来 +5.4 个点**。
- 这是整个仓库里最能说明"我懂为什么"的实验：同一份代码只改一个开关，结果就分开了。

## q4 · numpy 手写 MLP

`q4-numpy-mlp/numpy-mlp.ipynb`

- 只有 numpy：激活函数、softmax、交叉熵、反向传播全部手写，训练 100 epoch。
- 训练过程 Loss 0.3451 → 0.2761，测试集 **93.33%**。

## q5 · 手写卷积（im2col）

`q5-im2col-conv/im2col-convolution.ipynb`

- 实现 `im2col` / `col2im`，把滑窗卷积转成矩阵乘法，卷积和池化都不用框架。
- 每个张量形状都有断言检查，避免"能跑但算错"。
- 5 epoch 到 **97.77%**，每 epoch 约 22.5 秒——手写实现的开销是看得见的。

## q6 · IMDb 文本分类

`q6-imdb-text/imdb-text-classification.ipynb`

- 自己写 `build_vocab`、tokenizer、`Dataset`/`DataLoader`，直接读 aclImdb 的原始 txt。
- 训练 10 epoch，保存最优 checkpoint：验证 0.8538 → 测试 **0.8480**，F1 0.8503。

## 结果图

每个实验目录下的 `results/` 是从 notebook 输出里直接导出的训练曲线与结果图（共 20 张），顺序与代码执行顺序一致。例如：

| | |
|---|---|
| ![MLP 实验](q1-mlp/results/mlp-from-scratch_fig01.png) | ![MLP 对比](q1-mlp/results/mlp-from-scratch_fig04.png) |
| ![ResNet 对照](q3-resnet18/results/resnet18-from-scratch_fig02.png) | ![IMDb 训练](q6-imdb-text/results/imdb-text-classification_fig01.png) |

## 环境

```bash
pip install torch torchvision numpy matplotlib scikit-learn tensorflow
```

数据集用 `torchvision.datasets` / `tensorflow_datasets` 自动下载，notebook 里已写好。`q6` 首次运行会下 aclImdb（约 80MB）。
