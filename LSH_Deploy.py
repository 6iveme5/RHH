import numpy as np
import tensorflow as tf
import config
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from LSH import LSHConfidenceKMeans  # 需要在 LSH.py 中定义 `LSHConfidence`
from dataset_loader import load_dataset

# 1. 加载训练好的模型
model = tf.keras.models.load_model(config.TS_MODEL_SAVE_PATH)
print("成功加载模型:", config.TS_MODEL_SAVE_PATH)

# 2. 加载数据集
X_test, y_test = load_dataset(config.DATASET)

# 归一化
X_test = X_test / 255.0

# 3. 进行预测
y_probs = model.predict(X_test)
y_pred = np.argmax(y_probs, axis=1)

# 4. 计算 LSH 置信度分数
lsh_confidence = LSHConfidenceKMeans(X_test, y_pred)
confidence_scores = lsh_confidence.compute_scores(X_test, y_test)  # 传入 y_test

# 5. 按置信度排序
sorted_indices = np.argsort(confidence_scores)
sorted_confidence_scores = confidence_scores[sorted_indices]
sorted_y_test = y_test[sorted_indices]
sorted_y_pred = y_pred[sorted_indices]

# 6. 计算不同分位数的精度（precision）
percentiles = np.linspace(0, 100, 21)  # 设置 21 个分位数点
precisions = []

for percentile in percentiles:
    # 计算该分位数对应的索引
    percentile_index = int(len(sorted_confidence_scores) * (percentile / 100))

    # 获取该分位数的样本
    percentile_y_test = sorted_y_test[:percentile_index]
    percentile_y_pred = sorted_y_pred[:percentile_index]

    if len(percentile_y_test) > 0:
        # 计算该区间的加权精度
        precision = precision_score(percentile_y_test, percentile_y_pred, average='weighted')
        precisions.append(precision)
    else:
        precisions.append(0)  # 如果样本为空，则精度设为 0

# 7. 绘制曲线
plt.figure(figsize=(8, 6))
plt.plot(percentiles, precisions, 'm-', label="LSH Precision", linewidth=2)
plt.xlabel("Percentile Level")
plt.ylabel("Precision")
plt.title("Precision vs. Percentile for LSH")
plt.legend()
plt.grid()
plt.show()
