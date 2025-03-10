import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans


class LSHConfidenceKMeans:
    def __init__(self, X_train, y_train, num_planes=1000, k=10, w_same=1.0, w_diff=0.5, num_clusters=75):  # 将 num_clusters 设置为 75
        """
        基于 K-means 生成超平面的 LSH 置信度评估类
        :param X_train: 训练数据
        :param y_train: 训练标签
        :param num_planes: 生成的哈希超平面数量
        :param k: 最近邻个数
        :param w_same: 同类样本距离的权重
        :param w_diff: 异类样本距离的权重
        :param num_clusters: K-means 聚类的簇数量
        """
        self.num_planes = num_planes
        self.k = k
        self.w_same = w_same
        self.w_diff = w_diff
        self.num_clusters = num_clusters  # 调整簇的数量
        self.X_train = X_train
        self.y_train = y_train
        self.planes = None
        self.fit(X_train, y_train)

    def fit(self, X, y):
        """ 训练 LSH 哈希模型，使用 K-means 生成超平面 """
        self.X_train = X.reshape(X.shape[0], -1)
        self.y_train = y

        # 执行 K-means 聚类
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        kmeans.fit(self.X_train)
        cluster_centers = kmeans.cluster_centers_

        # 生成超平面（选取聚类中心对）
        num_centers = len(cluster_centers)
        planes = []
        for _ in range(self.num_planes):
            i, j = np.random.choice(num_centers, 2, replace=False)
            normal_vector = cluster_centers[i] - cluster_centers[j]  # 计算超平面的法向量
            normal_vector /= np.linalg.norm(normal_vector)  # 归一化
            planes.append(normal_vector)

        self.planes = np.array(planes)

    def transform(self, X):
        """ 计算哈希编码 """
        X = X.reshape(X.shape[0], -1)
        dot_product = np.dot(X, self.planes.T)
        hash_codes = np.sign(dot_product)
        return (hash_codes + 1) // 2  # 转换为 0/1 编码

    def compute_scores(self, X_test, y_test):
        """ 计算测试样本的置信度 """
        if self.X_train is None or self.y_train is None:
            raise ValueError("模型未训练，请先调用 `fit()` 方法")

        X_test_hash = self.transform(X_test)
        X_train_hash = self.transform(self.X_train)

        # 计算所有测试样本到训练样本的汉明距离
        distances = pairwise_distances(X_test_hash, X_train_hash, metric='hamming')

        confidence_scores = []

        for i in range(len(X_test)):
            # 获取当前测试样本的汉明距离
            sample_distances = distances[i]

            # 获取与当前测试样本同类别的索引和不同类别的索引
            same_class_mask = (self.y_train == y_test[i])
            diff_class_mask = (self.y_train != y_test[i])

            # 计算同类最近 k 个样本的最近邻距离
            same_class_distances = sample_distances[same_class_mask]
            if len(same_class_distances) > 0:
                d_same = np.sort(same_class_distances)[:self.k].min()  # 最近邻
            else:
                d_same = 1.0  # 没有同类数据，设为最大可能距离

            # 计算不同类最近 k 个样本的最近邻距离
            diff_class_distances = sample_distances[diff_class_mask]
            if len(diff_class_distances) > 0:
                d_diff = np.sort(diff_class_distances)[:self.k].min()
            else:
                d_diff = 1.0  # 没有不同类数据，设为最大可能距离

            # 计算加权后的置信度
            epsilon = 1e-6  # 防止除零
            confidence = (self.w_diff * d_diff) / (self.w_same * d_same + epsilon)
            confidence_scores.append(confidence)

        return np.array(confidence_scores)
    def get_score(self, X_test, y_test):
        """ 使用 compute_scores 计算得分 """
        return self.compute_scores(X_test, y_test)