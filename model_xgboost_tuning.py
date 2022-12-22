import math
import warnings
import xgboost
import numpy as np

import matplotlib.pyplot as plt

from xgboost import XGBClassifier


from data_load import load_total_data
from sklearn.model_selection import StratifiedKFold
# from utils import *

from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE

class XGBoostHT:
    def __init__(self):
        self.total_data, self.total_label, self.philips_data, self.philips_label, self.siemens_data, self.siemens_label = load_total_data()

        self.kfold = StratifiedKFold(n_splits=10)

    def feature_selection(self):
        rfe = RFE(estimator=XGBClassifier(eval_metric="auc"), n_features_to_select=20)
        rfe.fit(self.total_data, self.total_label)

        self.total_data = rfe.transform(self.total_data)

    def total(self):
        self.feature_selection()

        model = XGBClassifier(
            eval_metric="auc",
            # colsample_bytree=0.8,
            # gamma=0.5,
            # learning_rate=0.01,
            # max_depth=3,
            # min_child_weight=10,
            # n_estimators=200,
            # random_state=42,
            # subsample=0.6
        )

        n_iter = 0
        mean_fpr = np.linspace(0, 1, 100)
        tpers = []
        aucs = []
        cv_accuracy = []

        for train_idx, test_idx in self.kfold.split(self.total_data, self.total_label):
            x_train, x_test = self.total_data.iloc[train_idx], self.total_data.iloc[test_idx]
            y_train, y_test = self.total_label.iloc[train_idx], self.total_label.iloc[test_idx]

            model.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=200, verbose=False)

            fold_pred_train = model.score(x_train, y_train)
            pred = model.predict(x_test)
            pred_proba = model.predict_proba(x_test)[:, 1]

            fper, tper, threshold = roc_curve(self.total_label.iloc[test_idx].values.ravel(), pred_proba)
            tpers.append(np.interp(mean_fpr, fper, tper))
            roc_auc = auc(fper, tper)
            aucs.append(roc_auc)
            n_iter += 1

            plt.plot(fper, tper, lw=2, alpha=0.3, label="ROC fold %d (AUC = %2f)" % (n_iter, roc_auc))

            accuracy = np.round(accuracy_score(y_test, pred), 4)
            print(f"{n_iter}-fold 교차검증 정확도 : {accuracy}, 학습 정확도 {fold_pred_train}")

            cv_accuracy.append(accuracy)

        print(f"평균 검증 정확도: {np.mean(cv_accuracy)}")
        std_mean = np.std(cv_accuracy)

        upper = np.mean(cv_accuracy) + 1.96 * std_mean / math.sqrt(10)
        lower = np.mean(cv_accuracy) - 1.96 * std_mean / math.sqrt(10)

        print(f"학습 정확도 upper: {upper}, lower: {lower} (95% CI)")

        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black")

        mean_tpr = np.mean(tpers, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)

        print(f"평균 검증 AUC: {mean_auc}")

        std = np.std(aucs)
        upper = mean_auc + 1.96 * std / math.sqrt(10)
        lower = mean_auc - 1.96 * std / math.sqrt(10)
        print(f"AUC upper: {upper}, lower: {lower} (95% CI)")

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
        plt.title("ROC - XGBoost")
        plt.legend(loc="lower right")
        plt.show()


warnings.filterwarnings("ignore")


if __name__ == "__main__":
    mod = XGBoostHT()
    mod.total()
