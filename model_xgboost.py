import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_curve, auc
from xgboost import XGBClassifier
from data_load import load_data
from plot_roc import plot_roc_curve


class XGBoost:
    model: XGBClassifier

    def __init__(self, data: pd.DataFrame, label: pd.DataFrame):
        self.data = data
        self.label = label

        self.model = XGBClassifier(
            n_estimators=300,
            n_jobs=4,
            gamma=1.5,
            subsample=0.8,
            colsample_bytree=0.9,
            colsample_bylevel=0.9,
            reg_lambda=0.3,
            random_state=1823346,
            learning_rate=0.01,
            min_child_weight=10,
        )
        self.kfold = StratifiedKFold(n_splits=10)
        self.cv_accuracy = []
        self.selected_columns = []
        self.tpers = []
        self.fpers = []
        self.aucs = []

    def feature_selection(self):
        rfe = RFE(self.model, n_features_to_select=25)
        rfe.fit(self.data, self.label)

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
            accuracy = np.round(accuracy_score(y_test, fold_pred), 4)
            print(f"{n_iter}-fold ???????????? ????????? : {accuracy}, ?????? ????????? : {fold_pred_train}")

            self.cv_accuracy.append(accuracy)

        print(f"?????? ?????? ????????? : {np.mean(self.cv_accuracy)}")

        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black")

        mean_tpr = np.mean(self.tpers, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)

        print(f"?????? AUC : {mean_auc}")
        # J = mean_tpr - mean_fpr
        # idx = np.argmax(J)
        # best_threshold =

        plt.plot(
            mean_fpr,
            mean_tpr,
            color="blue",
            label="Mean ROC (AUC = %0.2f )" % mean_auc,
            lw=2,
            alpha=1,
        )

        try:
            with open("model.txt", "rb") as f:
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
            with open("model.txt", "wb") as f:
                pickle.dump({"model": self.model, "accuracy": mean_auc}, f)
            print("best updated!")

    def feature_importance(self):
        feature_importance = self.model.feature_importances_
        print("*********importance************")
        for idx, im in enumerate(feature_importance):
            print(f"{self.data.columns[idx]} : {im}")


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    data, label = load_data()
    xgb = XGBoost(data, label)
    xgb.train()
    # xgb.feature_importance()
    # xgb.feature_selection()
