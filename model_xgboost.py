import pickle

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


class XGBoost:
    def __init__(self, data, label):
        self.data = data
        self.label = label

        self.model = XGBClassifier()
        self.kfold = StratifiedKFold(n_splits=10)
        self.cv_accuracy = []

    def train(self):

        n_iter = 0

        for train_idx, test_idx in self.kfold.split(self.data, self.label):
            x_train, x_test = self.data[train_idx], self.data[test_idx]
            y_train, y_test = self.label[train_idx], self.label[test_idx]

            self.model.fit(x_train, y_train)

            fold_pred = self.model.predict(x_test)

            n_iter += 1

            accuracy = np.round(accuracy_score(y_test, fold_pred), 4)
            print(
                f"{n_iter} 교차검증 정확도 : {accuracy}, 학습 데이터 크기: {x_train.shape[0]}, 검증 데이터 크기: {x_test.shape[0]} "
            )

            self.cv_accuracy.append(accuracy)

        print(f"평균 검증 정확도 : {np.mean(self.cv_accuracy)}")
        try:
            with open("model.txt", "rb") as f:
                accuracy = pickle.load(f).get("accuracy")
        except:
            accuracy = -1

        if np.mean(self.cv_accuracy) > accuracy:
            with open("model.txt", "rb") as f:
                pickle.dump(
                    {"model": self.model, "accuracy": np.mean(self.cv_accuracy)}, f
                )
