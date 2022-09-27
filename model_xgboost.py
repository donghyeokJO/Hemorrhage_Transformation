import pickle
import warnings
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_curve, auc
from xgboost import XGBClassifier
from data_load import load_data, load_data_new


class XGBoost:
    data: pd.DataFrame
    label: pd.DataFrame
    model: XGBClassifier
    kfold: StratifiedKFold
    rfe: RFE

    def __init__(self, pca: bool):
        self.data, self.label = load_data(pca=pca)
        # self.data = self.data.loc[:, self.data.columns.isin(selected_columns)]

        self.model = XGBClassifier(
            n_estimators=300,
            n_jobs=4,
            gamma=0.02,
            subsample=0.6,
            colsample_bytree=0.9,
            colsample_bylevel=0.9,
            reg_lambda=1,
            random_state=8501372767,
            learning_rate=0.014,
            min_child_weight=10,
        )
        self.kfold = StratifiedKFold(n_splits=10)
        self.cv_accuracy = []
        self.selected_columns = []
        self.tpers = []
        self.fpers = []
        self.aucs = []
        self.pca = pca

    def feature_selection(self):
        rfe = RFE(self.model, n_features_to_select=20, verbose=True)
        rfe.fit(self.data, self.label, eval_metric="auc")

        self.selected_columns = [
            col_nm for idx, col_nm in enumerate(self.data.columns) if rfe.support_[idx]
        ]

        print(self.selected_columns)

    def train(self):
        n_iter = 0
        mean_fpr = np.linspace(0, 1, 100)

        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111, aspect="equal")

        for train_idx, test_idx in self.kfold.split(self.data, self.label):
            # print(train_idx, test_idx)
            x_train, x_test = self.data.iloc[train_idx], self.data.iloc[test_idx]
            y_train, y_test = self.label.iloc[train_idx], self.label.iloc[test_idx]

            self.model.fit(x_train, y_train, eval_metric="auc")

            fold_pred_train = self.model.score(x_train, y_train)
            fold_pred = self.model.predict(x_test)
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

            # J = tper - fper
            # idx = np.argmax(J)
            # best_threshold = threshold[idx]
            # sens, spec = tper[idx], 1 - fper[idx]
            # print(
            #     "%d-Fold Best threshold = %.3f, Sensitivity = %.3f, Specificity = %.3f"
            #     % (n_iter, best_threshold, sens, spec)
            # )

            accuracy = np.round(accuracy_score(y_test, fold_pred), 4)
            print(f"{n_iter}-fold 교차검증 정확도 : {accuracy}, 학습 정확도 : {fold_pred_train}")

            self.cv_accuracy.append(accuracy)

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
        # J = mean_tpr - mean_fpr
        # idx = np.argmax(J)
        # best_threshold =

        # plt.plot(
        #     mean_fpr,
        #     mean_tpr,
        #     color="blue",
        #     label="Mean ROC (AUC = %0.2f )" % mean_auc,
        #     lw=2,
        #     alpha=1,
        # )

        try:
            if self.pca:
                with open("model_xgboost_pca.txt", "rb") as f:
                    accuracy = pickle.load(f).get("accuracy")
            else:
                with open("model_xgboost.txt", "rb") as f:
                    accuracy = pickle.load(f).get("accuracy")

        except:
            accuracy = -1

        print(f"current best: {accuracy}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.legend(loc="lower right")
        plt.show()

        if mean_auc > accuracy:
            if self.pca:
                with open("model_xgboost_pca.txt", "wb") as f:
                    pickle.dump({"model": self.model, "accuracy": mean_auc}, f)
            else:
                with open("model_xgboost.txt", "wb") as f:
                    pickle.dump({"model": self.model, "accuracy": mean_auc}, f)

            print("best updated!")

        return mean_auc

    def feature_importance(self):
        feature_importance = self.model.feature_importances_

        feature_importance_dict = {}
        feature_importance_dict2 = {}
        for idx, im in enumerate(feature_importance):
            if "HU" in self.data.columns[idx]:
                feature_importance_dict[self.data.columns[idx]] = im
            feature_importance_dict2[self.data.columns[idx]] = im

        print(
            sorted(feature_importance_dict2.items(), key=lambda x: x[1], reverse=True)
        )

        # x = [i for i in range(0, 80)]
        x = feature_importance_dict.keys()
        y = feature_importance_dict.values()
        # fig = plt.figure(figsize=[12, 12])
        plt.plot(x, y, color="red", alpha=0.3)
        plt.xlabel("HU")
        plt.ylabel("Feature Importance")
        plt.title("Feature Importance - XGBoost")
        plt.show()

    def test(self):
        print("********** Test Phase **********")
        if self.pca:
            with open("model_xgboost_pca.txt", "rb") as f:
                model = pickle.load(f).get("model")

        else:
            with open("model_xgboost.txt", "rb") as f:
                model = pickle.load(f).get("model")

        test_data, test_label = load_data_new(pca=self.pca)
        predict_ = model.predict(test_data)
        accuracy = np.round(accuracy_score(test_label, predict_), 4)
        print(f"정확도: {accuracy}")

        pred_proba = model.predict_proba(test_data)[:, 1]

        fper, tper, threshold = roc_curve(test_label.values.ravel(), pred_proba)

        auc_score = auc(fper, tper)

        print(f"평균 AUC : {auc_score}")


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    xgb = XGBoost(pca=False)
    xgb.train()
    xgb.test()
    # xgb.feature_importance()
    # xgb.feature_selection()
