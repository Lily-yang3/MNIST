# MNIST 手写数字识别（PyTorch 实现）

一个基于 **PyTorch + 自定义 MNIST 解析** 的深度学习项目。  
本项目完整展示了从 **读取原生 IDX 格式数据 → 构建 Dataset → 定义模型 → 训练 → 保存模型 → 测试** 的完整流程。

**项目亮点：**

- 不使用 `torchvision.datasets.MNIST`，而是手动解析 IDX 数据集  
- 自定义 `Dataset` & `DataLoader`  
- 实现一个简单的 MLP 神经网络  
- 支持 CPU / GPU / Apple MPS 自动切换  
- 训练脚本与测试脚本分离  
- 工程结构清晰，适合作为 PyTorch 入门模版  

---

## 📂 项目结构

```bash
MNIST/
│
├── data/                      # MNIST 原始数据（idx、gz）
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
│
├── program/
│   ├── mnist_dataset.py       # 自定义 Dataset + DataLoader（解析 IDX）
│   ├── model.py               # MLP 模型结构 SimpleMLP
│   ├── utils.py               # 训练与评估函数：train_one_epoch / eval_model
│   ├── train_mnist.py         # 训练脚本：只在训练集上训练，保存 mnist_mlp.pth
│   └── test_mnist.py          # 测试脚本：加载模型，在测试集上评估
│
├── mnist_mlp.pth              # 训练好的模型权重（通常通过 .gitignore 忽略）
├── .gitignore
└── README.md
🛠️ 环境配置

建议使用 Conda 创建独立环境：

conda create -n mnist-env python=3.10
conda activate mnist-env


安装依赖：

pip install torch torchvision numpy pillow matplotlib tqdm


可选：检查 GPU（或 Apple MPS）是否可用：

import torch
print("CUDA:", torch.cuda.is_available())
print("MPS:", torch.backends.mps.is_available())

📥 下载 MNIST 数据集

MNIST 官方下载地址（IDX 格式）：

https://storage.googleapis.com/cvdf-datasets/mnist/

需要下载以下 4 个 .gz 文件，并放入项目中的 data/ 目录下：

train-images-idx3-ubyte.gz

train-labels-idx1-ubyte.gz

t10k-images-idx3-ubyte.gz

t10k-labels-idx1-ubyte.gz

项目中已内置解压逻辑，首次运行前可执行：

cd MNIST
python program/mnist_dataset.py


如果解压成功，会看到类似输出：

[INFO] 解压: train-images-idx3-ubyte.gz -> train-images-idx3-ubyte
[DONE] MNIST 解压完成！

🧱 代码模块说明
1. 数据集与 DataLoader（mnist_dataset.py）

手动解析 MNIST 的 IDX 文件（图像与标签）

实现 read_idx_images 与 read_idx_labels

封装为 MNISTIdxDataset(Dataset)，返回 (image, label)

提供：

train_loader  # 训练集 DataLoader
test_loader   # 测试集 DataLoader

2. 模型（model.py）

一个简单的多层感知机（MLP）结构：

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # [B, 1, 28, 28] -> [B, 784]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)             # logits [B, 10]
        return x

3. 训练与评估辅助函数（utils.py）

train_one_epoch(model, optimizer, criterion, train_loader, device, epoch)

在训练集上跑一整轮（一个 epoch）

每 100 个 batch 打印一次平均损失

eval_model(model, criterion, test_loader, device)

切换到 model.eval() 模式

不计算梯度（torch.no_grad()）

返回：avg_loss, acc

4. 训练脚本（train_mnist.py）

自动选择设备（MPS / CUDA / CPU）

创建模型、损失函数、优化器

在训练集上训练若干个 epoch

训练完毕后保存模型权重到 mnist_mlp.pth

🚀 训练模型

在项目根目录执行：

cd MNIST
python program/train_mnist.py


你将会看到类似输出（示例）：

Using device: mps
Epoch [1] Step [100/938] Loss: 0.8833
Epoch [1] Step [200/938] Loss: 0.4134
...
Epoch [5] Step [900/938] Loss: 0.0744
训练完成，模型已保存为 mnist_mlp.pth


训练过程中 不会使用测试集，以避免“偷看”测试集，保持评估的严格性。

🔍 测试模型

训练完成后，在项目根目录运行：

python program/test_mnist.py


该脚本会：

加载 mnist_mlp.pth

在 test_loader（测试集）上前向计算

输出测试集的平均 Loss 与 Accuracy

示例输出：

Using device: mps
模型权重已加载：mnist_mlp.pth
Test Loss: 0.0567
Test Accuracy: 98.15%
