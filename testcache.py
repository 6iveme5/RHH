import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import config
import trustscore
import trustscore_evaluation
from LSH import LSHConfidenceKMeans
import dataset_loader  # 导入 dataset_loader.py

# 根据 config.py 中的 DATASET 参数加载数据集
dataset_name = config.DATASET  # 获取 config.py 中的 DATASET 参数

# 使用 dataset_loader 中的 load_dataset 方法加载训练集和测试集
X_train, y_train = dataset_loader.load_dataset(dataset_name, split="train")
X_test, y_test = dataset_loader.load_dataset(dataset_name, split="test")

# 将图像数据从 (num_samples, height, width, channels) 转换为 (num_samples, height * width * channels)
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# 将 LSHConfidenceKMeans 集成到实验中
extra_plot_title = f"{dataset_name} | Logistic Regression | Predict Correct"
percentile_levels = [0 + 0.5 * i for i in range(200)]
signal_names = ["Trust Score", "LSH Confidence"]
signals = [trustscore.TrustScore(), LSHConfidenceKMeans(X_train_flattened[:1300], y_train[:1300])]  # 使用训练集数据
trainer = trustscore_evaluation.run_logistic
trustscore_evaluation.run_precision_recall_experiment_general(X_train_flattened,
                                                            y_train,
                                                            n_repeats=10,
                                                            percentile_levels=percentile_levels,
                                                            trainer=trainer,
                                                            signal_names=signal_names,
                                                            signals=signals,
                                                            extra_plot_title=extra_plot_title,
                                                            skip_print=True,
                                                            predict_when_correct=True)

# 进行不正确预测的实验
extra_plot_title = f"{dataset_name} | Logistic Regression | Predict Incorrect"
percentile_levels = [70 + 0.5 * i for i in range(60)]
trustscore_evaluation.run_precision_recall_experiment_general(X_train_flattened,
                                                            y_train,
                                                            n_repeats=10,
                                                            percentile_levels=percentile_levels,
                                                            trainer=trainer,
                                                            signal_names=signal_names,
                                                            signals=signals,
                                                            extra_plot_title=extra_plot_title,
                                                            skip_print=True)
