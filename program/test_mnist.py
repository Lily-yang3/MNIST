# test_mnist.py

import torch
import torch.nn as nn

from mnist_dataset import test_loader      # 测试集的 DataLoader
from model import SimpleMLP                # 模型结构
from utils import eval_model               # 测试函数


def get_device():
    """自动选择合适的设备：MPS > CUDA > CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main():
    # 1. 选择设备
    device = get_device()
    print("Using device:", device)

    # 2. 创建模型并加载训练好的权重
    model = SimpleMLP().to(device)
    model.load_state_dict(torch.load("mnist_mlp.pth", map_location=device))
    print("模型权重已加载：mnist_mlp.pth")

    # 3. 损失函数（和训练时相同）
    criterion = nn.CrossEntropyLoss()

    # 4. 在测试集上评估
    test_loss, test_acc = eval_model(model, criterion, test_loader, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
