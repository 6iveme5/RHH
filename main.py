import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader

# 1. 加载 CIFAR-10 数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 需要 224x224 的输入
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 训练集
train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. 采用 ResNet 作为特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的分类层

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)  # 展平为 (batch_size, feature_dim)

# 初始化 ResNet 特征提取器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = FeatureExtractor().to(device)
feature_extractor.eval()  # 设为评估模式，不更新参数

# 3. 生成随机超平面
def generate_hyperplanes(feature_dim, num_hash_bits):
    """
    生成随机超平面
    feature_dim: 特征维度
    num_hash_bits: 生成的哈希码长度
    """
    return np.random.randn(num_hash_bits, feature_dim)  # 服从标准正态分布

# 4. 计算超平面哈希码
def compute_hash_codes(features, hyperplanes):
    """
    计算哈希码
    features: (N, D) 矩阵，N 为样本数，D 为特征维度
    hyperplanes: (K, D) 矩阵，K 为哈希码长度
    """
    hash_codes = np.sign(features @ hyperplanes.T)  # 计算点积并取符号
    return (hash_codes > 0).astype(int)  # 1 代表正，0 代表负

# 5. 计算 CIFAR-10 训练集的哈希码
num_hash_bits = 32  # 设定哈希码长度
hyperplanes = None  # 初始化超平面矩阵
hash_codes_list = []

with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(device)
        features = feature_extractor(images).cpu().numpy()  # 提取特征

        if hyperplanes is None:
            feature_dim = features.shape[1]
            hyperplanes = generate_hyperplanes(feature_dim, num_hash_bits)

        hash_codes = compute_hash_codes(features, hyperplanes)
        hash_codes_list.append(hash_codes)

# 拼接所有哈希码
hash_codes = np.vstack(hash_codes_list)
print("哈希码形状:", hash_codes.shape)  # (训练样本数, num_hash_bits)
np.save('hash_codes.npy', hash_codes)  # 将哈希码保存为 .npy 文件
