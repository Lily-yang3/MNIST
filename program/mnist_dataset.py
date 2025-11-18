import torch
from tqdm import tqdm
import matplotlib
import gzip
import os
from pathlib import Path
import shutil
import torchvision
from torchvision import transforms
import numpy as np
import struct
from torch.utils.data import Dataset, DataLoader
from PIL import Image
device = "cuda:0" if torch.cuda.is_available() else "cpu"
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

def unzip_mnist():
    gz_files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for fname in gz_files:
        gz_path = DATA_DIR / fname
        out_path = DATA_DIR / fname[:-3]

        if not gz_path.exists():
            print(f"[WARN] 找不到文件: {gz_path}")
            continue

        if out_path.exists():
            print(f"[SKIP] 已存在: {out_path}")
            continue

        print(f"[INFO] 解压: {gz_path} -> {out_path}")
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    print("[DONE] MNIST 解压完成！")

if __name__ == "__main__":
    print(">>> 现在在 __main__ 分支里，准备调用 unzip_mnist()")
    unzip_mnist()
def read_idx_images(path):
    with open(path, "rb") as f:
        data = f.read()

    # 前 16 字节：magic, num_images, rows, cols
    magic, num_images, rows, cols = struct.unpack_from(">IIII", data, 0)
    assert magic == 2051, f"Magic number 不对，期望 2051，实际 {magic}"

    # 剩下的是像素，uint8
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    images = images.reshape(num_images, rows, cols)
    return images  # shape: [N, 28, 28]


# 读标签
def read_idx_labels(path):
    with open(path, "rb") as f:
        data = f.read()

    magic, num_labels = struct.unpack_from(">II", data, 0)
    assert magic == 2049, f"Magic number 不对，期望 2049，实际 {magic}"

    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return labels  # shape: [N]
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MNISTIdxDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        # 用你自己写的函数读文件
        self.images = read_idx_images(images_path)   # shape: [N, 28, 28], uint8
        self.labels = read_idx_labels(labels_path)   # shape: [N], uint8

        assert len(self.images) == len(self.labels), "图片数量和标签数量不一致！"

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]              # numpy, [28, 28]
        label = int(self.labels[idx])       # 转成 Python int

        # ToTensor/Normalize 一般希望输入 PIL Image 或者 tensor
        if self.transform is not None:
            # 灰度图，用 'L' 模式
            img = Image.fromarray(img, mode="L")
            img = self.transform(img)
        else:
            # 不用 transform 的话，手动转 tensor 并归一化到 [0,1]
            img = torch.from_numpy(img).unsqueeze(0).float() / 255.0

        return img, label
# 构造路径
train_images_path = DATA_DIR / "train-images-idx3-ubyte"
train_labels_path = DATA_DIR / "train-labels-idx1-ubyte"
test_images_path  = DATA_DIR / "t10k-images-idx3-ubyte"
test_labels_path  = DATA_DIR / "t10k-labels-idx1-ubyte"

# 实例化数据集
train_dataset = MNISTIdxDataset(train_images_path, train_labels_path, transform=transform)
test_dataset  = MNISTIdxDataset(test_images_path,  test_labels_path,  transform=transform)

# 再封装成 DataLoader
batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)
if __name__ == "__main__":
    # 先只测试数据
    images, labels = next(iter(train_loader))
    print(images.shape, labels.shape)  # 预期: torch.Size([64, 1, 28, 28]) torch.Size([64])


