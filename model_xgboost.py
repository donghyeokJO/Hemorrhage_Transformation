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
from data_load import load_total_data
from utils import *


class XGBoost:
    data: pd.DataFrame
    label: pd.DataFrame
    model: XGBClassifier
    kfold: StratifiedKFold
    rfe: RFE

    def __init__(self, pca: bool):
        (
            self.total_data,
            self.total_label,
            self.philips_data,
            self.philips_label,
            self.siemens_data,
            self.siemens_label,
        ) = load_total_data()

        self.save_path = "XGB_total_train_pca.pkl" if pca else "XGB_total_train.pkl"
        # self.save_path = (
        #     "XGB_total_train_pca_added.pkl" if pca else "XGB_total_train_added.pkl"
        # )

        self.save_path_siemens = (
            "XGB_siemens_total_train_pca.pkl" if pca else "XGB_siemens_total_train.pkl"
        )
        self.save_path_philips = (
            "XGB_philips_total_train_pca.pkl" if pca else "XGB_philips_total_train.pkl"
        )

        try:
            with open(self.save_path, "rb") as f:
                pkl = pickle.load(f)
                best_params_ = pkl.get("best_params_")
                self.model = XGBClassifier(**best_params_)
        except:
            self.model = XGBClassifier(
                colsample_bylevel=0.7,
                colsample_bytree=0.7,
                eval_metric="auc",
                gamma=0.02,
                learning_rate=0.01,
                min_child_weight=10,
                n_estimators=300,
                n_jobs=4,
                random_state=42,
                reg_lambda=1,
                subsample=0.8,
            )

        self.kfold = StratifiedKFold(n_splits=10)
        self.cv_accuracy = []
        self.selected_columns = []
        self.tpers = []
        self.fpers = []
        self.aucs = []
        self.pca = pca

        self.params = {
            "min_child_weight": [1, 5, 10],
            "gamma": [0.5, 1, 1.5],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.1, 0.01],
            "n_estimators": [100, 200, 300],
            "random_state": [42],
            "eval_metric": ["auc"],
        }

    def feature_selection(self):
        model = XGBClassifier()
        rfe = RFE(model, n_features_to_select=20, verbose=False)
        rfe.fit(self.total_data, self.total_label, eval_metric="auc")

        self.selected_columns = [
            col_nm for idx, col_nm in enumerate(self.data.columns) if rfe.support_[idx]
        ]

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

    def total_train_fold(self):
        total_data, total_label = self.total_data, self.total_label

        train_data, test_data, train_label, test_label = train_test_split(
            total_data, total_label, test_size=0.2, random_state=42
        )

        model = XGBClassifier
        params = {
            "min_child_weight": [1, 5, 10],
            "gamma": [0.5, 1, 1.5],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.1, 0.01],
            "n_estimators": [100, 200, 300],
            "random_state": [42],
            "eval_metric": ["auc"],
        }

        grid_search_func(train_data, train_label, model, params, self.save_path)

        with open(self.save_path, "rb") as f:
            info = pickle.load(f)

        model = XGBClassifier(**info.get("best_params_"))
        model.fit(train_data, train_label)

        pred = model.predict(test_data)
        pred_proba = model.predict_proba(test_data)[:, 1]

        accuracy = np.round(accuracy_score(test_label, pred), 4)
        fper, tper, threshold = roc_curve(test_label.values.ravel(), pred_proba)
        auc_score = auc(fper, tper)

        print(f"테스트 정확도: {accuracy}")
        print(f"테스트 mAUC: {auc_score}")
        with open("xg_total.pkl", "wb") as f:
            pickle.dump({"acc": accuracy, "mauc": auc_score}, f)

    def test_total(self):
        fig = plt.figure(figsize=[12, 12])
        ax = fig.add_subplot(111, aspect="equal")
        tpers = []
        aucs = []
        cv_accuracy = []

        kfold = StratifiedKFold(n_splits=10)

        with open(self.save_path, "rb") as f:
            info = pickle.load(f)

        model = XGBClassifier(**info.get("best_params_"))
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

            model.fit(x_train, y_train, verbose=False)

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
        plt.title("ROC - XGBoost")
        plt.legend(loc="lower right")
        plt.show()

    def train_siemens(self):
        train_data, test_data, train_label, test_label = train_test_split(
            self.siemens_data, self.siemens_label, test_size=0.2, random_state=42
        )

        model = XGBClassifier

        grid_search_func(
            train_data, train_label, model, self.params, self.save_path_siemens
        )

        with open(self.save_path_siemens, "rb") as f:
            info = pickle.load(f)

        model = XGBClassifier(**info.get("best_params_"))
        model.fit(train_data, train_label)

        pred = model.predict(test_data)
        pred_proba = model.predict_proba(test_data)[:, 1]

        accuracy = np.round(accuracy_score(test_label, pred), 4)
        fper, tper, threshold = roc_curve(test_label.values.ravel(), pred_proba)
        auc_score = auc(fper, tper)

        print(f"siemens 테스트 정확도: {accuracy}")
        print(f"siemens 테스트 mAUC: {auc_score}")
        with open("xg_siemens.pkl", "wb") as f:
            pickle.dump({"acc": accuracy, "mauc": auc_score}, f)

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
        model = XGBClassifier(**info.get("best_params_"))

        for train_idx, test_idx in self.kfold.split(
            self.siemens_data, self.siemens_label
        ):
            # print(train_idx, test_idx)
            x_train, x_test = (
                self.siemens_data.iloc[train_idx],
                self.siemens_data.iloc[test_idx],
            )
            y_train, y_test = (
                self.siemens_label.iloc[train_idx],
                self.siemens_label.iloc[test_idx],
            )

            model.fit(x_train, y_train, verbose=False)

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
        plt.title("ROC - XGBoost")
        plt.legend(loc="lower right")
        plt.show()

    def train_philips(self):
        train_data, test_data, train_label, test_label = train_test_split(
            self.philips_data, self.philips_label, test_size=0.1, random_state=42
        )

        model = XGBClassifier

        grid_search_func(
            train_data, train_label, model, self.params, self.save_path_philips
        )

        with open(self.save_path_philips, "rb") as f:
            info = pickle.load(f)

        model = XGBClassifier(**info.get("best_params_"))
        model.fit(train_data, train_label)

        pred = model.predict(test_data)
        pred_proba = model.predict_proba(test_data)[:, 1]

        accuracy = np.round(accuracy_score(test_label, pred), 4)
        fper, tper, threshold = roc_curve(test_label.values.ravel(), pred_proba)
        auc_score = auc(fper, tper)

        print(f"philips 테스트 정확도: {accuracy}")
        print(f"philips 테스트 mAUC: {auc_score}")
        with open("xg_philips.pkl", "wb") as f:
            pickle.dump({"acc": accuracy, "mauc": auc_score}, f)

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
        model = XGBClassifier(**info.get("best_params_"))

        for train_idx, test_idx in self.kfold.split(
            self.philips_data, self.philips_label
        ):
            # print(train_idx, test_idx)
            x_train, x_test = (
                self.philips_data.iloc[train_idx],
                self.philips_data.iloc[test_idx],
            )
            y_train, y_test = (
                self.philips_label.iloc[train_idx],
                self.philips_label.iloc[test_idx],
            )

            model.fit(x_train, y_train, verbose=False)

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
        plt.title("ROC - XGBoost")
        plt.legend(loc="lower right")
        plt.show()


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    cat = XGBoost(pca=False)
    print("total")
    cat.total_train_fold()
    print("total_test")
    cat.test_total()
    # print("siemens")
    # cat.train_siemens()
    # print("siemens_test")
    # cat.siemens()
    # print("philips")
    # cat.train_philips()
    # print("philips_test")
    # cat.philips()
    # xgb.total_train()
    # xgb.train()
    # xgb.test()
    # xgb.feature_importance()
    # xgb.feature_selection()
