import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
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

        self.model = CatBoostClassifier()
        self.kfold = StratifiedKFold(n_splits=10)
        self.cv_accuracy = []
        self.tpers = []
        self.fpers = []
        self.aucs = []

    def train(self):
        n_iter = 0
        mean_fpr = np.linspace(0, 1, 100)

        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111, aspect="equal")

        for train_idx, test_idx in self.kfold.split(self.data, self.label):
            # print(train_idx, test_idx)
            x_train, x_test = self.data.iloc[train_idx], self.data.iloc[test_idx]
            y_train, y_test = self.label.iloc[train_idx], self.label.iloc[test_idx]

            self.model.fit(x_train, y_train, verbose=False)

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
            print(f"{n_iter}-fold 교차검증 정확도 : {accuracy}, 학습 정확도 : {fold_pred_train}")

            self.cv_accuracy.append(accuracy)

        print(f"평균 검증 정확도 : {np.mean(self.cv_accuracy)}")

        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black")

        mean_tpr = np.mean(self.tpers, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)

        print(f"평균 AUC : {mean_auc}")

        plt.plot(
            mean_fpr,
            mean_tpr,
            color="blue",
            label="Mean ROC (AUC = %0.2f )" % mean_auc,
            lw=2,
            alpha=1,
        )

        try:
            with open("model_catboost.txt", "rb") as f:
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
            with open("model_catboost.txt", "wb") as f:
                pickle.dump({"model": self.model, "accuracy": mean_auc}, f)
            print("best updated!")


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    data, label = load_data()
    cat = CatBoost(data, label)
    cat.train()
