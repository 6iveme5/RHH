import tensorflow as tf
import numpy as np
from scipy.io import loadmat


def load_mnist():
    """
    加载 MNIST 数据集并进行预处理
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = X_train[..., np.newaxis], X_test[..., np.newaxis]  # 增加通道维度
    X_train, X_test = X_train / 255.0, X_test / 255.0  # 归一化
    return (X_train, y_train), (X_test, y_test)


def load_cifar10():
    """
    加载 CIFAR-10 数据集并进行预处理
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()  # 修正 y_train 和 y_test 的形状
    X_train, X_test = X_train / 255.0, X_test / 255.0  # 归一化
    return (X_train, y_train), (X_test, y_test)


def load_svhn():
    """
    加载 SVHN 数据集并进行预处理
    """
    data = loadmat('train_32x32.mat')  # 加载 SVHN 数据
    X_train = data['X']
    y_train = data['y']
    X_train = np.transpose(X_train, (3, 0, 1, 2))  # 转换为 [num_samples, height, width, channels]
    y_train = y_train.flatten()

    data_test = loadmat('test_32x32.mat')  # 加载 SVHN 测试数据
    X_test = data_test['X']
    y_test = data_test['y']
    X_test = np.transpose(X_test, (3, 0, 1, 2))  # 转换为 [num_samples, height, width, channels]
    y_test = y_test.flatten()

    X_train, X_test = X_train / 255.0, X_test / 255.0  # 归一化
    return (X_train, y_train), (X_test, y_test)


def load_cifar100():
    """
    加载 CIFAR-100 数据集并进行预处理
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
    y_train, y_test = y_train.flatten(), y_test.flatten()  # 修正 y_train 和 y_test 的形状
    X_train, X_test = X_train / 255.0, X_test / 255.0  # 归一化
    return (X_train, y_train), (X_test, y_test)


def load_dataset(dataset_name, split="test"):
    """
    根据数据集名称加载数据
    split: "train" 或 "test" 用来选择加载训练数据还是测试数据
    """
    if dataset_name == "MNIST":
        (X_train, y_train), (X_test, y_test) = load_mnist()
    elif dataset_name == "CIFAR10":
        (X_train, y_train), (X_test, y_test) = load_cifar10()
    elif dataset_name == "SVHN":
        (X_train, y_train), (X_test, y_test) = load_svhn()
    elif dataset_name == "CIFAR100":
        (X_train, y_train), (X_test, y_test) = load_cifar100()
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    # 根据传入的参数返回训练集或测试集
    if split == "train":
        return X_train, y_train
    elif split == "test":
        return X_test, y_test
    else:
        raise ValueError(f"Split {split} is not supported. Choose 'train' or 'test'.")

