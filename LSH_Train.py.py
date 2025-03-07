import numpy as np
import tensorflow as tf
import config  # 导入超参数配置文件

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


# 训练并保存模型
def train_and_save_model(X_train, y_train):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1) if config.DATASET == "MNIST" else (32, 32, 3)),
        tf.keras.layers.Dense(config.dense_units, activation='relu'),
        tf.keras.layers.Dense(config.num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size,
              validation_split=config.validation_split)

    # 保存训练好的模型
    model.save(config.LSH_MODEL_SAVE_PATH)
    print(f"Model saved to {config.LSH_MODEL_SAVE_PATH}")
    return model


# 主函数
if __name__ == "__main__":
    # 加载数据
    X_train, y_train, _, _ = load_data()

    # 训练并保存模型
    train_and_save_model(X_train, y_train)
