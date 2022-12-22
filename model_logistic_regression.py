import pickle
import warnings
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc
from data_load import load_data, load_data_philips, load_data_new, load_total_data
from utils import *


class LogiReg:
    data: pd.DataFrame
    label: pd.DataFrame
    model: LogisticRegression
    kfold: StratifiedKFold

    def __init__(self, pca):
        self.total_data, self.total_label, self.philips_data, self.philips_label, self.siemens_data, self.siemens_label = load_total_data()
        self.pca = pca

        self.kfold = StratifiedKFold(n_splits=10)
        self.cv_accuracy = []
        self.tpers = []
        self.fpers = []
        self.aucs = []

        self.save_path = "lr_total_train_pca.pkl" if pca else "lr_total_train.pkl"

        self.save_path_siemens = (
            "lr_siemens_total_train_pca.pkl" if pca else "lr_siemens_total_train.pkl"
        )
        self.save_path_philips = (
            "lr_philips_total_train_pca.pkl" if pca else "lr_philips_total_train.pkl"
        )

        C = np.logspace(-4, 4, 50)
        penalty = ["l1", "l2"]
        solver = ["liblinear"]

        self.params = dict(C=C, penalty=penalty, solver=solver)

    def total_train_fold(self):
        train_data, test_data, train_label, test_label = train_test_split(
            self.total_data, self.total_label, test_size=0.1, random_state=42
        )

        model = LogisticRegression

        grid_search_func(train_data, train_label, model, self.params, self.save_path)

        with open(self.save_path, "rb") as f:
            info = pickle.load(f)

        model = LogisticRegression(**info.get("best_params_"))
        model.fit(train_data, train_label)

        pred = model.predict(test_data)
        pred_proba = model.predict_proba(test_data)[:, 1]

        accuracy = np.round(accuracy_score(test_label, pred), 4)
        fper, tper, threshold = roc_curve(test_label.values.ravel(), pred_proba)
        auc_score = auc(fper, tper)

        print(f"정확도: {accuracy}")
        print(f"mAUC: {auc_score}")

    def test_total(self):
        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111, aspect="equal")
        tpers = []
        aucs = []
        cv_accuracy = []

        kfold = StratifiedKFold(n_splits=10)

        with open(self.save_path, "rb") as f:
            info = pickle.load(f)

        model = LogisticRegression(**info.get("best_params_"))
        n_iter = 0
        mean_fpr = np.linspace(0, 1, 100)

        for train_idx, test_idx in kfold.split(self.total_data, self.total_label):
            x_train, x_test = (
                self.total_data.iloc[train_idx],
                self.total_data.iloc[test_idx],
            )
            y_train, y_test = (
                self.total_label.iloc[train_idx],
                self.total_label.iloc[test_idx],
            )

            model.fit(x_train, y_train)

            train = model.score(x_train, y_train)
            pred = model.predict(x_test)
            pred_proba = model.predict_proba(x_test)[:, 1]

            fper, tper, threshold = roc_curve(
                self.total_label.iloc[test_idx].values.ravel(), pred_proba
            )
            tpers.append(np.interp(mean_fpr, fper, tper))
            roc_auc = auc(fper, tper)
            aucs.append(roc_auc)
            n_iter += 1

            plt.plot(
                fper,
                tper,
                lw=2,
                alpha=0.3,
                label="ROC fold %d (AUC = %.2f)" % (n_iter, roc_auc),
            )

            accuracy = np.round(accuracy_score(y_test, pred), 4)
            print(f"{n_iter}-fold 교차검증 정확도 : {accuracy}, 학습 정확도 : {train}")
            cv_accuracy.append(accuracy)

        print(f"평균 검증 정확도 : {np.mean(cv_accuracy)}")

        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black")

        mean_tpr = np.mean(tpers, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)

        print(f"평균 AUC : {mean_auc}")
        import math

        print(np.mean(aucs))
        std = np.std(aucs)
        upper = mean_auc + 1.96 * std / math.sqrt(10)
        lower = mean_auc - 1.96 * std / math.sqrt(10)
        print(f"upper: {upper}, lower: {lower} (95% CI)")

        std_mean = np.std(cv_accuracy)

        upper = np.mean(cv_accuracy) + 1.96 * std_mean / math.sqrt(10)
        lower = np.mean(cv_accuracy) - 1.96 * std_mean / math.sqrt(10)

        print(f"학습 정확도 upper: {upper}, lower: {lower} (95% CI)")

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
        plt.title("ROC - Logistic Regression")
        plt.legend(loc="lower right")
        plt.show()

    def train_siemens(self):
        train_data, test_data, train_label, test_label = train_test_split(
            self.siemens_data, self.siemens_label, test_size=0.2, random_state=42
        )

        model = LogisticRegression

        grid_search_func(
            train_data, train_label, model, self.params, self.save_path_siemens
        )

        with open(self.save_path_siemens, "rb") as f:
            info = pickle.load(f)

        model = LogisticRegression(**info.get("best_params_"))
        model.fit(train_data, train_label)

        pred = model.predict(test_data)
        pred_proba = model.predict_proba(test_data)[:, 1]

        accuracy = np.round(accuracy_score(test_label, pred), 4)
        fper, tper, threshold = roc_curve(test_label.values.ravel(), pred_proba)
        auc_score = auc(fper, tper)

        print(f"정확도: {accuracy}")
        print(f"mAUC: {auc_score}")

    def siemens(self):
        n_iter = 0
        mean_fpr = np.linspace(0, 1, 100)

        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111, aspect="equal")
        tpers = []
        aucs = []
        cv_accuracy = []

        with open(self.save_path_siemens, "rb") as f:
            info = pickle.load(f)

        print(info.get("best_params_"))
        model = LogisticRegression(**info.get("best_params_"))

        for train_idx, test_idx in self.kfold.split(self.siemens_data, self.siemens_label):
            # print(train_idx, test_idx)
            x_train, x_test = self.siemens_data.iloc[train_idx], self.siemens_data.iloc[test_idx]
            y_train, y_test = self.siemens_label.iloc[train_idx], self.siemens_label.iloc[test_idx]

            model.fit(x_train, y_train)

            fold_pred_train = model.score(x_train, y_train)
            fold_pred = model.predict(x_test)
            fold_pred_proba = model.predict_proba(x_test)[:, 1]

            fper, tper, threshold = roc_curve(
                self.siemens_label.iloc[test_idx].values.ravel(), fold_pred_proba
            )
            tpers.append(np.interp(mean_fpr, fper, tper))
            roc_auc = auc(fper, tper)
            aucs.append(roc_auc)
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

            cv_accuracy.append(accuracy)

        print(f"평균 검증 정확도 : {np.mean(cv_accuracy)}")

        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black")

        mean_tpr = np.mean(tpers, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)

        print(f"평균 AUC : {mean_auc}")
        import math

        print(np.mean(aucs))
        std = np.std(aucs)
        upper = mean_auc + 1.96 * std / math.sqrt(10)
        lower = mean_auc - 1.96 * std / math.sqrt(10)
        print(f"upper: {upper}, lower: {lower} (95% CI)")

        std_mean = np.std(cv_accuracy)

        upper = np.mean(cv_accuracy) + 1.96 * std_mean / math.sqrt(10)
        lower = np.mean(cv_accuracy) - 1.96 * std_mean / math.sqrt(10)

        print(f"학습 정확도 upper: {upper}, lower: {lower} (95% CI)")

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
        plt.title("ROC - Logistic Regression")
        plt.legend(loc="lower right")
        plt.show()

    def train_philips(self):
        train_data, test_data, train_label, test_label = train_test_split(
            self.philips_data, self.philips_label, test_size=0.2, random_state=42
        )

        model = LogisticRegression
        grid_search_func(
            train_data, train_label, model, self.params, self.save_path_philips
        )

        with open(self.save_path_philips, "rb") as f:
            info = pickle.load(f)

        model = LogisticRegression(**info.get("best_params_"))
        model.fit(train_data, train_label)

        pred = model.predict(test_data)
        pred_proba = model.predict_proba(test_data)[:, 1]

        accuracy = np.round(accuracy_score(test_label, pred), 4)
        fper, tper, threshold = roc_curve(test_label.values.ravel(), pred_proba)
        auc_score = auc(fper, tper)

        print(f"정확도: {accuracy}")
        print(f"mAUC: {auc_score}")

    def philips(self):
        n_iter = 0
        mean_fpr = np.linspace(0, 1, 100)

        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111, aspect="equal")
        tpers = []
        aucs = []
        cv_accuracy = []

        with open(self.save_path_philips, "rb") as f:
            info = pickle.load(f)

        print(info.get("best_params_"))
        model = LogisticRegression(**info.get("best_params_"))

        for train_idx, test_idx in self.kfold.split(
            self.philips_data, self.philips_label
        ):
            # print(train_idx, test_idx)
            x_train, x_test = self.philips_data.iloc[train_idx], self.philips_data.iloc[test_idx]
            y_train, y_test = self.philips_label.iloc[train_idx], self.philips_label.iloc[test_idx]

            model.fit(x_train, y_train)

            fold_pred_train = model.score(x_train, y_train)
            fold_pred = model.predict(x_test)
            fold_pred_proba = model.predict_proba(x_test)[:, 1]

            fper, tper, threshold = roc_curve(
                self.philips_label.iloc[test_idx].values.ravel(), fold_pred_proba
            )
            tpers.append(np.interp(mean_fpr, fper, tper))
            roc_auc = auc(fper, tper)
            aucs.append(roc_auc)
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

            cv_accuracy.append(accuracy)

        print(f"평균 검증 정확도 : {np.mean(cv_accuracy)}")

        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black")

        mean_tpr = np.mean(tpers, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)

        print(f"평균 AUC : {mean_auc}")
        import math

        print(np.mean(aucs))
        std = np.std(aucs)
        upper = mean_auc + 1.96 * std / math.sqrt(10)
        lower = mean_auc - 1.96 * std / math.sqrt(10)
        print(f"upper: {upper}, lower: {lower} (95% CI)")

        std_mean = np.std(cv_accuracy)

        upper = np.mean(cv_accuracy) + 1.96 * std_mean / math.sqrt(10)
        lower = np.mean(cv_accuracy) - 1.96 * std_mean / math.sqrt(10)

        print(f"학습 정확도 upper: {upper}, lower: {lower} (95% CI)")

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
        plt.title("ROC - Logistic Regression")
        plt.legend(loc="lower right")
        plt.show()


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    cat = LogiReg(pca=False)
    print("total")
    cat.total_train_fold()
    print("total_test")
    cat.test_total()
    print("siemens")
    cat.train_siemens()
    print("siemens_test")
    cat.siemens()
    print("philips")
    cat.train_philips()
    print("philips_test")
    cat.philips()
