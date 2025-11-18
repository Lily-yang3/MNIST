# train_mnist.py

import torch
import torch.nn as nn
import torch.optim as optim

from mnist_dataset import train_loader, test_loader
from model import SimpleMLP
from utils import train_one_epoch, eval_model


def get_device():
    """选择合适的设备：优先 MPS(CPU), 其次 CUDA，最后 CPU。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")      # Mac 上的 GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")     # Nvidia GPU
    else:
        return torch.device("cpu")


def main():
    # 1. 设备
    device = get_device()
    print("Using device:", device)

    # 2. 创建模型 / 损失 / 优化器
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 5

    # 3. 只训练，不做测试集评估
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, optimizer, criterion, train_loader, device, epoch)

    # 4. 保存最终模型
    torch.save(model.state_dict(), "mnist_mlp.pth")
    print("训练完成，模型已保存为 mnist_mlp.pth")


if __name__ == "__main__":
    main()






