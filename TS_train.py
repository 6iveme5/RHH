import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import config  # 导入超参数配置
from dataset_loader import load_dataset

# 加载数据集
X_train, y_train = load_dataset(config.DATASET, split="train")
X_test, y_test = load_dataset(config.DATASET, split="test")


# 归一化
X_train, X_test = X_train / 255.0, X_test / 255.0

# 划分训练和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=config.validation_split, random_state=42
)

# 打印数据形状以确保正确
print(f"训练数据形状: {X_train.shape}, 验证数据形状: {X_val.shape}, 测试数据形状: {X_test.shape}")

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

# 构建并训练模型
model = build_model(X_train.shape[1:], config.num_classes)
history = model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_val, y_val))

# 评估模型
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")

# 保存模型
model.save(config.TS_MODEL_SAVE_PATH)
print("模型已保存到 "+ config.TS_MODEL_SAVE_PATH)
