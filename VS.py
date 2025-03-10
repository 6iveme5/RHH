import numpy as np
import tensorflow as tf
import config
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from LSH import LSHConfidenceKMeans  # 需要在 LSH.py 中定义 `LSHConfidence`
from dataset_loader import load_dataset
from trustscore import TrustScore  # 需要有 TrustScore 的实现
from model_confidence import ModelConfidence  # 导入 ModelConfidence 类

# 1. 加载训练好的模型
model = tf.keras.models.load_model(config.TS_MODEL_SAVE_PATH)
print("成功加载模型:", config.TS_MODEL_SAVE_PATH)

# 2. 加载数据集
X_test, y_test = load_dataset(config.DATASET)

# 归一化
X_test = X_test / 255.0

# 3. 进行预测
# 根据数据集调整输入的形状
if config.DATASET == "MNIST":
    # MNIST 数据是 28x28 的灰度图像，输入形状应为 (num_samples, 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
elif config.DATASET == "CIFAR10" or config.DATASET == "CIFAR100":
    # CIFAR-10 和 CIFAR-100 是 32x32 彩色图像，输入形状应为 (num_samples, 32, 32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
elif config.DATASET == "SVHN":
    # SVHN 数据是 32x32 彩色图像，输入形状应为 (num_samples, 32, 32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)

y_probs = model.predict(X_test)
y_pred = np.argmax(y_probs, axis=1)

# 4. 计算 LSH 置信度分数
lsh_confidence = LSHConfidenceKMeans(X_test, y_pred)
lsh_scores = lsh_confidence.compute_scores(X_test, y_test)

# 5. 计算 TrustScore 置信度分数
trust_model = TrustScore()
# 如果数据是 MNIST（形状: (num_samples, 28, 28, 1)），展平为 (num_samples, 28*28)
if config.DATASET == "MNIST":
    X_test = X_test.reshape(X_test.shape[0], -1)  # 变成 (10000, 784)
elif config.DATASET == "CIFAR10" or config.DATASET == "CIFAR100":
    X_test = X_test.reshape(X_test.shape[0], -1)  # 变成 (10000, 3072)

trust_model.fit(X_test, y_test)
trust_scores = trust_model.get_score(X_test, y_pred)

# 6. 计算 ModelConfidence 置信度分数
model_confidence = ModelConfidence(model, X_test, y_test)
model_conf_scores = model_confidence.compute_confidence(X_test, y_test)

# 7. 计算不同分位数的精度
percentiles = np.linspace(0, 100, 21)
lsh_precisions = []
trust_precisions = []
model_conf_precisions = []

for scores, precisions in zip([lsh_scores, trust_scores, model_conf_scores],
                              [lsh_precisions, trust_precisions, model_conf_precisions]):
    sorted_indices = np.argsort(scores)
    sorted_y_test = y_test[sorted_indices]
    sorted_y_pred = y_pred[sorted_indices]
    for percentile in percentiles:
        percentile_index = int(len(scores) * (percentile / 100))
        percentile_y_test = sorted_y_test[:percentile_index]
        percentile_y_pred = sorted_y_pred[:percentile_index]
        if len(percentile_y_test) > 0:
            precision = precision_score(percentile_y_test, percentile_y_pred, average='weighted')
            precisions.append(precision)
        else:
            precisions.append(0)

# 8. 绘制曲线
plt.figure(figsize=(8, 6))
plt.plot(percentiles, lsh_precisions, 'm-', label="LSH Precision", linewidth=2)
plt.plot(percentiles, trust_precisions, 'b-', label="TrustScore Precision", linewidth=2)
plt.plot(percentiles, model_conf_precisions, 'g-', label="Model Confidence Precision", linewidth=2)
plt.xlabel("Percentile Level")
plt.ylabel("Precision")
plt.title("Precision vs. Percentile for LSH, TrustScore, and Model Confidence")
plt.legend()
plt.grid()
plt.show()
