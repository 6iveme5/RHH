import numpy as np
import tensorflow as tf
from sklearn.metrics import pairwise_distances, roc_curve, auc
import matplotlib.pyplot as plt
import config  # 导入超参数配置文件
from sklearn.preprocessing import label_binarize

# 随机超平面哈希类（LSH方法）
class RandomHyperplaneHashing:
    def __init__(self, num_planes=1000):
        self.num_planes = num_planes

    def fit(self, X):
        self.X_train = X
        n_samples, n_features = X.shape
        # 生成随机超平面（随机法向量）
        self.random_planes = np.random.randn(self.num_planes, n_features)

    def transform(self, X):
        dot_product = np.dot(X, self.random_planes.T)
        hash_codes = np.sign(dot_product)  # 计算哈希码
        return hash_codes

    def get_reliability_score(self, X_test, y_test, X_train=None, y_train=None):
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train

        # 获取测试数据的哈希码
        X_test_hash = self.transform(X_test)
        # 获取训练数据的哈希码
        X_train_hash = self.transform(X_train)

        # 计算测试数据与训练数据的汉明距离
        distances = pairwise_distances(X_test_hash, X_train_hash, metric='hamming')

        # 计算每个测试样本的可靠性分数
        reliability_scores = []
        for i in range(len(X_test)):
            same_label_indices = np.where(y_train == y_test[i])[0]
            # 计算同标签样本的汉明距离
            same_label_distances = distances[i, same_label_indices]
            reliability_scores.append(np.mean(same_label_distances))  # 使用均值作为可靠性分数

        return np.array(reliability_scores)


# 数据加载和预处理
def load_data(dataset_name=config.DATASET):
    if dataset_name == "MNIST":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train, X_test = X_train[..., np.newaxis] / 255.0, X_test[..., np.newaxis] / 255.0
    elif dataset_name == "CIFAR10":
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X_train, X_test = X_train / 255.0, X_test / 255.0
        y_train, y_test = y_train.flatten(), y_test.flatten()
    return X_train, y_train, X_test, y_test


# 绘制多分类ROC曲线
def plot_roc_curve(y_test, y_pred, reliability_scores):
    # 对标签进行二值化处理
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    # 创建图形用于绘制ROC曲线
    plt.figure(figsize=(10, 8))

    # 绘制每个类的ROC曲线
    for i in range(y_test_bin.shape[1]):
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], reliability_scores)

        # 计算AUC
        roc_auc = auc(fpr, tpr)

        # 绘制第i类的ROC曲线
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

    # 绘制随机分类器（对角线）
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # 设置图标标签
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('多分类ROC曲线')
    plt.legend(loc='lower right')

    # 显示图像
    plt.show()


# 测试并评估模型
# 测试并评估模型
def evaluate_model(X_test, y_test, X_train, y_train):
    # 加载已保存的模型
    model = tf.keras.models.load_model(config.TS_MODEL_SAVE_PATH)

    # 使用训练好的模型对测试集进行预测
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # 初始化并训练随机超平面哈希
    rh_hashing = RandomHyperplaneHashing(num_planes=config.num_planes)
    rh_hashing.fit(X_train.reshape(X_train.shape[0], -1))

    # 计算测试集的可靠性分数
    reliability_scores = rh_hashing.get_reliability_score(X_test.reshape(X_test.shape[0], -1), y_test,
                                                          X_train.reshape(X_train.shape[0], -1), y_train)

    # 输出第一个测试样本的可靠性分数
    print(f"Sample 0 Reliability Score: {reliability_scores[0]:.4f}")

    # 计算精度随分位数变化
    sorted_indices = np.argsort(reliability_scores)  # 按照可靠性评分从小到大排序
    sorted_reliability_scores = reliability_scores[sorted_indices]
    sorted_y_test = y_test[sorted_indices]
    sorted_y_pred = y_pred[sorted_indices]

    # 分位数区间设置
    percentiles = np.linspace(0, 100, 11)  # 创建10个分位数区间
    precisions = []

    for p in percentiles:
        # 计算对应分位数的样本数
        idx = int(len(sorted_reliability_scores) * p / 100)
        selected_indices = sorted_indices[:idx+1]  # 选择该分位数区间的样本
        correct_predictions = np.sum(sorted_y_test[selected_indices] == sorted_y_pred[selected_indices])
        precision = correct_predictions / len(selected_indices)  # 计算精度
        precisions.append(precision)

    # 绘制精度随分位数变化的图像
    plt.plot(percentiles, precisions, color='green', linestyle='-', marker='o', label="LSH (Precision)")
    plt.xlabel('Percentile Level')
    plt.ylabel('Precision')
    plt.title(f'Precision vs Percentile for {config.DATASET} Dataset')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 绘制可靠性分数分布
    plt.hist(reliability_scores, bins=20, alpha=0.7, label="Reliability Score")
    plt.xlabel("Reliability Score")
    plt.ylabel("Frequency")
    plt.title(f"Reliability Score Distribution on Test Data\n"
              f"Num Planes: {config.num_planes}, Dataset: {config.DATASET}, Epoch:{config.epochs}")
    plt.legend()
    plt.show()

    # 绘制多分类AUROC图像
    plot_roc_curve(y_test, y_pred, reliability_scores)

# 主函数
if __name__ == "__main__":
    # 加载数据
    X_train, y_train, X_test, y_test = load_data()
    evaluate_model(X_test, y_test, X_train, y_train)
