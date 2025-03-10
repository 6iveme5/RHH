import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

from dataset_loader import load_dataset
from trustscore import TrustScore
import config  # 导入超参数配置

# 加载已训练的模型
model = tf.keras.models.load_model(config.TS_MODEL_SAVE_PATH)
print("已成功加载模型")

# 加载测试数据
X_test, y_test = load_dataset(config.DATASET)

# 归一化
X_test = X_test / 255.0

# 进行预测
y_pred = np.argmax(model.predict(X_test), axis=1)

# 计算 Trust Score
ts = TrustScore()
ts.fit(X_test.reshape(X_test.shape[0], -1), y_test)
trust_scores = ts.get_score(X_test.reshape(X_test.shape[0], -1), y_pred)

# 按 Trust Score 排序数据
sorted_indices = np.argsort(trust_scores)
sorted_trust_scores = trust_scores[sorted_indices]
sorted_y_test = y_test[sorted_indices]
sorted_y_pred = y_pred[sorted_indices]

# 计算不同分位数的精度
percentiles = np.linspace(0, 100, 21)  # 设置分位数区间，从 0 到 100，分为 20 个区间
precisions = []

for percentile in percentiles:
    # 计算分位数对应的索引
    percentile_index = int(len(sorted_trust_scores) * (percentile / 100))

    # 获取该分位数区间的样本
    percentile_y_test = sorted_y_test[:percentile_index]
    percentile_y_pred = sorted_y_pred[:percentile_index]

    # 计算该区间的分类精度
    precision = precision_score(percentile_y_test, percentile_y_pred, average='weighted')
    precisions.append(precision)

# 绘制精度随分位数变化的趋势，风格与LSH图相同
plt.figure(figsize=(8, 6))
plt.plot(percentiles, precisions, 'm-', label="Trust Score Precision", linewidth=2)
plt.xlabel("Percentile Level")
plt.ylabel("Precision")
plt.title("Precision vs. Percentile for Trust Score")
plt.legend()
plt.grid()
plt.show()
