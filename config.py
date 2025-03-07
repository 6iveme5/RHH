# config.py - 超参数配置文件

# 数据集选择
DATASET = "CIFAR10"  # 可选 "CIFAR10", "MNIST"

# 模型超参数
batch_size = 64
epochs = 100
learning_rate = 0.001
num_classes = 10  # 分类数
validation_split = 0.1  # 训练集划分多少用于验证集

# CNN 结构
conv_filters = [32, 64, 64]  # 每层 CNN 过滤器数
kernel_size = (3, 3)  # 卷积核大小
pool_size = (2, 2)  # 池化层大小
dense_units = 64  # 全连接层神经元个数

# 随机超平面哈希配置
num_planes = 100  # 随机超平面数

# 模型保存路径
LSH_MODEL_SAVE_PATH = 'model/LSH_saved_model.h5'
TS_MODEL_SAVE_PATH= 'model/TS_saved_model.h5'