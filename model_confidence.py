import numpy as np


class ModelConfidence:
    def __init__(self, model, X_train, y_train):
        """
        使用模型的预测概率计算模型置信度。
        :param model: 训练好的 Keras 模型
        :param X_train: 训练数据
        :param y_train: 训练标签
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def compute_confidence(self, X_test, y_test):
        """
        计算模型对测试数据的置信度。
        :param X_test: 测试数据
        :param y_test: 测试标签
        :return: 置信度分数
        """
        # 获取模型的预测概率
        y_probs = self.model.predict(X_test)

        # 计算最大概率值作为模型的置信度
        confidence_scores = np.max(y_probs, axis=1)

        return confidence_scores
