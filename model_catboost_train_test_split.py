import pickle
import warnings
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from catboost import CatBoostClassifier
from data_load import load_data


class CatBoost:
    data: pd.DataFrame
    label: pd.DataFrame
    model: CatBoostClassifier
    kfold: StratifiedKFold

    def __init__(self, data: pd.DataFrame, labels: pd.DataFrame):
        self.data = data
        self.label = labels

        self.model = CatBoostClassifier(
            learning_rate=0.001,
            reg_lambda=1,
            random_state=1011101,
            subsample=0.5,
        )

        (
            self.train_data,
            self.test_data,
            self.train_label,
            self.test_label,
        ) = train_test_split(self.data, self.label, test_size=0.2, random_state=42)
        self.cv_accuracy = []
        self.tpers = []
        self.fpers = []
        self.aucs = []
        self.kfold = StratifiedKFold(n_splits=10)
        self.test_accuracy = []

    def train(self):
        n_iter = 0
        mean_fpr = np.linspace(0, 1, 100)

        plt.figure(figsize=[12, 12])
        # ax = fig.add_subplot(111, aspect="equal")

        self.model.fit(self.train_data, self.train_label, verbose=False)

        pred_train = self.model.score(self.train_data, self.train_label)
        pred = self.model.predict(self.test_data)
        pred_proba = self.model.predict_proba(self.test_data)[:, 1]
        # print(pred_proba)

        fper, tper, threshold = roc_curve(self.test_label.values.ravel(), pred_proba)

        roc_auc = auc(fper, tper)

        accuracy = np.round(accuracy_score(self.test_label, pred), 4)
        print(f"정확도 : {accuracy}, 학습 정확도 : {pred_train}")

        J = tper - fper
        idx = np.argmax(J)
        best_threshold = threshold[idx]
        sens, spec = tper[idx], 1 - fper[idx]

        print(
            "Best threshold = %.3f, Sensitivity = %.3f, Specificity = %.3f"
            % (best_threshold, sens, spec)
        )

        plt.plot(
            fper,
            tper,
            color="blue",
            label="ROC (AUC = %.2f)" % roc_auc,
            lw=2,
            alpha=1,
        )
        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.legend(loc="lower right")
        plt.show()

    def train_kfold(self):
        n_iter = 0
        mean_fpr = np.linspace(0, 1, 100)

        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111, aspect="equal")

        for train_idx, test_idx in self.kfold.split(self.train_data, self.train_label):
            x_train, x_test = (
                self.train_data.iloc[train_idx],
                self.train_data.iloc[test_idx],
            )
            y_train, y_test = (
                self.train_label.iloc[train_idx],
                self.train_label.iloc[test_idx],
            )

            self.model.fit(x_train, y_train, verbose=False)

            fold_pred_train = self.model.score(x_train, y_train)
            fold_pred = self.model.predict(x_test)
            fold_pred_test = self.model.predict(self.test_data)
            fold_pred_proba = self.model.predict_proba(x_test)[:, 1]

            fper, tper, threshold = roc_curve(
                self.label.iloc[test_idx].values.ravel(), fold_pred_proba
            )

            self.tpers.append(np.interp(mean_fpr, fper, tper))
            roc_auc = auc(fper, tper)
            self.aucs.append(roc_auc)

            n_iter += 1

            plt.plot(
                fper,
                tper,
                lw=2,
                alpha=0.3,
                label="ROC fold %d (AUC = %.2f)" % (n_iter, roc_auc),
            )

            J = tper - fper
            idx = np.argmax(J)
            best_threshold = threshold[idx]
            sens, spec = tper[idx], 1 - fper[idx]

            accuracy = np.round(accuracy_score(y_test, fold_pred), 4)
            test_accuracy = np.round(accuracy_score(self.test_label, fold_pred_test), 4)

            print(
                f"{n_iter}-fold 교차검증 정확도 : {accuracy}, 학습 정확도 : {fold_pred_train} 테스트 정확도 : {test_accuracy}"
            )

            self.cv_accuracy.append(accuracy)
            self.test_accuracy.append(test_accuracy)

        print(f"평균 검증 정확도 : {np.mean(self.cv_accuracy)}")
        std_mean = np.std(self.cv_accuracy)
        upper = np.mean(self.cv_accuracy) + 1.96 * std_mean / math.sqrt(10)
        lower = np.mean(self.cv_accuracy) - 1.96 * std_mean / math.sqrt(10)

        print(f"학습 정확도 upper: {upper}, lower: {lower} (95% CI)")

        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black")

        mean_tpr = np.mean(self.tpers, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)

        print(f"평균 AUC : {mean_auc}")
        print(np.mean(self.aucs))
        std = np.std(self.aucs)
        upper = mean_auc + 1.96 * std / math.sqrt(10)
        lower = mean_auc - 1.96 * std / math.sqrt(10)
        print(f"upper: {upper}, lower: {lower} (95% CI)")

        plt.plot(
            mean_fpr,
            mean_tpr,
            color="blue",
            label="Mean ROC (AUC = %0.2f )" % mean_auc,
            lw=2,
            alpha=1,
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.legend(loc="lower right")
        plt.show()

        return mean_auc

    def feature_importance(self):
        feature_importance = self.model.feature_importances_

        feature_importance_dict = {}
        feature_importance_dict2 = {}

        for idx, im in enumerate(feature_importance):
            if "HU" in self.data.columns[idx]:
                feature_importance_dict[self.data.columns[idx]] = im
            print(f"{self.data.columns[idx]}: {im}")
            feature_importance_dict2[self.data.columns[idx]] = im

        print(
            sorted(feature_importance_dict2.items(), key=lambda x: x[1], reverse=True)
        )

        # x = feature_importance_dict.keys()
        x = [i for i in range(0, 80)]
        y = feature_importance_dict.values()

        # fig = plt.figure(figsize=[12, 12])
        plt.plot(x, y, color="red", alpha=0.3)
        plt.xlabel("HU")
        plt.ylabel("Feature Importance")
        plt.title("Feature Importance - CatBoost")
        plt.show()


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    data, label = load_data()
    cat = CatBoost(data, label)
    cat.train_kfold()
    # cat.feature_importance()
