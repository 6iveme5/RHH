import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from trustscore import TrustScore
import config  # 导入超参数配置

# 加载数据集
if config.DATASET == "MNIST":
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = X_train[..., np.newaxis], X_test[..., np.newaxis]  # 增加通道维度
elif config.DATASET == "CIFAR10":
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()  # 修正 flatten() 方式
    y_test = y_test.flatten()

# 归一化
X_train, X_test = X_train / 255.0, X_test / 255.0

# 划分训练和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=config.validation_split, random_state=42
)


def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(config.conv_filters[0], config.kernel_size, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(config.pool_size),
        layers.Conv2D(config.conv_filters[1], config.kernel_size, activation='relu'),
        layers.MaxPooling2D(config.pool_size),
        layers.Conv2D(config.conv_filters[2], config.kernel_size, activation='relu'),
        layers.Flatten(),
        layers.Dense(config.dense_units, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = build_model(X_train.shape[1:], config.num_classes)
history = model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_val, y_val))

# 评估模型
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")

# 保存模型
model.save(config.TS_MODEL_SAVE_PATH)
print("模型已保存到"+ config.TS_MODEL_SAVE_PATH)

# 计算 Trust Score
ts = TrustScore()
ts.fit(X_train.reshape(X_train.shape[0], -1), y_train)

trust_scores = ts.get_score(X_test.reshape(X_test.shape[0], -1), y_pred)

# 画出 Trust Score 分布，并添加 epoch 和数据集信息
plt.figure(figsize=(8, 6))
plt.hist(trust_scores, bins=20, alpha=0.7, label="Trust Score")
plt.xlabel("Trust Score")
plt.ylabel("Frequency")
plt.legend()
plt.title(f"Trust Score Distribution\nDataset: {config.DATASET}, Epochs: {config.epochs}")
plt.show()
