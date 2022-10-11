import math
import pickle
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_curve, auc
from lightgbm import LGBMClassifier
from data_load import load_data, load_data_new, load_data_philips
from utils import *


class LightGBM:
    data: pd.DataFrame
    label: pd.DataFrame
    model: LGBMClassifier
    kfold: StratifiedKFold

    def __init__(self, pca: bool):
        self.data, self.label = load_data(pca=pca)
        self.data_new, self.label_new = load_data_new(pca=pca)

        self.total_data = pd.concat([self.data, self.data_new], axis=0)
        self.total_label = pd.concat([self.label, self.label_new], axis=0)

        self.data_philips, self.label_philips = load_data_philips(pca=pca)

        self.pca = pca
        self.save_path = "LGBM_total_train_pca.pkl" if pca else "LGBM_total_train.pkl"

        self.save_path_siemens = (
            "LGBM_siemens_total_train_pca.pkl"
            if pca
            else "LGBM_siemens_total_train.pkl"
        )
        self.save_path_philips = (
            "LGBM_philips_total_train_pca.pkl"
            if pca
            else "LGBM_philips_total_train.pkl"
        )

        try:
            with open(self.save_path, "rb") as f:
                pkl = pickle.load(f)
                best_params_ = pkl.get("best_params_")
                self.model = LGBMClassifier(**best_params_)
        except:
            self.model = LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                objective="binary",
                learning_rate=0.12,
                reg_alpha=12.2,
                colsample_bytree=0.8,
                min_child_samples=20,
                min_child_weight=1e-5,
                subsample_freq=10,
                subsample=0.8,
            )

        self.kfold = StratifiedKFold(n_splits=10)
        self.cv_accuracy = []
        self.tpers = []
        self.fpers = []
        self.aucs = []
        self.selected_columns = []

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

            # J = tper - fper
            # idx = np.argmax(J)
            # best_threshold = threshold[idx]
            # sens, spec = tper[idx], 1 - fper[idx]
            # print(
            #     "%d-Fold Best threshold = %.3f, Sensitivity = %.3f, Specificity = %.3f"
            #     % (n_iter, best_threshold, sens, spec)
            # )

        print(f"평균 검증 정확도 : {np.mean(self.cv_accuracy)}")
        std_mean = np.std(self.cv_accuracy)
        import math

        upper = np.mean(self.cv_accuracy) + 1.96 * std_mean / math.sqrt(10)
        lower = np.mean(self.cv_accuracy) - 1.96 * std_mean / math.sqrt(10)

        print(f"학습 정확도 upper: {upper}, lower: {lower} (95% CI)")

        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black")

        mean_tpr = np.mean(self.tpers, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)

        print(f"평균 AUC : {mean_auc}")
        import math

        # print(np.mean(self.aucs))
        std = np.std(self.aucs)
        upper = mean_auc + 1.96 * std / math.sqrt(10)
        lower = mean_auc - 1.96 * std / math.sqrt(10)
        print(f"upper: {upper}, lower: {lower} (95% CI)")

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
                with open("model_lightgbm_pca.txt", "rb") as f:
                    accuracy = pickle.load(f).get("accuracy")
            else:
                with open("model_lightgbm.txt", "rb") as f:
                    accuracy = pickle.load(f).get("accuracy")

        except:
            accuracy = -1

        print(f"current best: {accuracy}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC - LightGBM")
        plt.legend(loc="lower right")
        plt.show()

        if mean_auc > accuracy:
            if self.pca:
                with open("model_lightgbm_pca.txt", "wb") as f:
                    pickle.dump({"model": self.model, "accuracy": mean_auc}, f)
            else:
                with open("model_lightgbm.txt", "wb") as f:
                    pickle.dump({"model": self.model, "accuracy": mean_auc}, f)

            print("best updated!")

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

        # x = [i for i in range(0, 80)]
        x = feature_importance_dict.keys()
        y = feature_importance_dict.values()
        # fig = plt.figure(figsize=[12, 12])
        plt.plot(x, y, color="red", alpha=0.3)
        plt.xlabel("HU")
        plt.ylabel("Feature Importance")
        plt.title("Feature Importance - LightGBM")
        plt.show()

    def feature_selection(self):
        model = LGBMClassifier()
        rfe = RFE(model, n_features_to_select=20, verbose=False)
        rfe.fit(self.total_data, self.total_label, eval_metric="auc")

        self.selected_columns = [
            col_nm for idx, col_nm in enumerate(self.data.columns) if rfe.support_[idx]
        ]

    def test(self):
        print("********** Test Phase **********")
        if self.pca:
            with open("model_lightgbm_pca.txt", "rb") as f:
                model = pickle.load(f).get("model")

        else:
            with open("model_lightgbm.txt", "rb") as f:
                model = pickle.load(f).get("model")

        test_data, test_label = load_data_new(pca=self.pca)
        predict_ = model.predict(test_data)
        accuracy = np.round(accuracy_score(test_label, predict_), 4)
        print(f"정확도: {accuracy}")

        pred_proba = model.predict_proba(test_data)[:, 1]

        fper, tper, threshold = roc_curve(test_label.values.ravel(), pred_proba)

        auc_score = auc(fper, tper)

        print(f"평균 AUC : {auc_score}")

    def total_train(self):
        total_data = self.total_data.loc[
            :, self.data.columns.isin(self.selected_columns)
        ]
        total_label = self.total_label

        train_data, test_data, train_label, test_label = train_test_split(
            total_data, total_label, test_size=0.1, random_state=42
        )

        self.model.fit(train_data, total_label)

        train_pred = self.model.predict(train_data)
        train_pred_proba = self.model.predict_proba(train_data)[:, 1]

        train_accuracy = np.round(accuracy_score(train_label, train_pred), 4)
        print(f"학습 정확도: {train_accuracy}")

        fper, tper, threshold = roc_curve(train_label.values.ravel(), train_pred_proba)
        auc_score = auc(fper, tper)
        print(f"학습 mAUC: {auc_score}")

        predict_ = self.model.predict(test_data)
        accuracy = np.round(accuracy_score(test_label, predict_), 4)
        print(f"정확도: {accuracy}")

        pred_proba = self.model.predict_proba(test_data)[:, 1]

        fper, tper, threshold = roc_curve(test_label.values.ravel(), pred_proba)

        auc_score = auc(fper, tper)

        print(f"mAUC: {auc_score}")

    def total_train_fold(self):
        # total_data, total_label = self.total_data, self.total_label
        # total_data, total_label = (
        #     self.total_data.loc[:, self.data.columns.isin(self.selected_columns)],
        #     self.total_label,
        # )

        train_data, test_data, train_label, test_label = train_test_split(
            self.total_data, self.total_label, test_size=0.1, random_state=42
        )

        model = LGBMClassifier
        params = {
            "n_estimators": [200, 300],
            "max_depth": [4, 6, 10],
            "objective": ["binary"],
            "learning_rate": [0.12, 0.1, 0.01],
            "reg_alpha": [12.2, 10, 8],
            "colsample_bytree": [0.8, 0.9, 1],
            "min_child_samples": [20],
            "min_child_weight": [1e-5],
            "subsample_freq": [10],
            "subsample": [0.8, 0.9, 1.0],
            "n_jobs": [10],
        }

        grid_search_func(train_data, train_label, model, params, self.save_path)

        with open(self.save_path, "rb") as f:
            info = pickle.load(f)

        model = LGBMClassifier(**info.get("best_params_"))
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

        model = LGBMClassifier(**info.get("best_params_"))
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
        plt.title("ROC - LGBM")
        plt.legend(loc="lower right")
        plt.show()

    def train_siemens(self):
        train_data, test_data, train_label, test_label = train_test_split(
            self.data, self.label, test_size=0.2, random_state=42
        )

        model = LGBMClassifier
        params = {
            "n_estimators": [200, 300],
            "max_depth": [4, 6, 10],
            "objective": ["binary"],
            "learning_rate": [0.12, 0.1, 0.01],
            "reg_alpha": [12.2, 10, 8],
            "colsample_bytree": [0.8, 0.9, 1],
            "min_child_samples": [20],
            "min_child_weight": [1e-5],
            "subsample_freq": [10],
            "subsample": [0.8, 0.9, 1.0],
            "n_jobs": [10],
        }

        grid_search_func(train_data, train_label, model, params, self.save_path_siemens)

        with open(self.save_path_siemens, "rb") as f:
            info = pickle.load(f)

        model = LGBMClassifier(**info.get("best_params_"))
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
        model = LGBMClassifier(**info.get("best_params_"))

        for train_idx, test_idx in self.kfold.split(self.data, self.label):
            # print(train_idx, test_idx)
            x_train, x_test = self.data.iloc[train_idx], self.data.iloc[test_idx]
            y_train, y_test = self.label.iloc[train_idx], self.label.iloc[test_idx]

            model.fit(x_train, y_train, verbose=False)

            fold_pred_train = model.score(x_train, y_train)
            fold_pred = model.predict(x_test)
            fold_pred_proba = model.predict_proba(x_test)[:, 1]

            fper, tper, threshold = roc_curve(
                self.label.iloc[test_idx].values.ravel(), fold_pred_proba
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
        plt.title("ROC - LGBM")
        plt.legend(loc="lower right")
        plt.show()

    def train_philips(self):
        train_data, test_data, train_label, test_label = train_test_split(
            self.data_philips, self.label_philips, test_size=0.2, random_state=42
        )

        model = LGBMClassifier
        params = {
            "n_estimators": [200, 300],
            "max_depth": [4, 6, 10],
            "objective": ["binary"],
            "learning_rate": [0.12, 0.1, 0.01],
            "reg_alpha": [12.2, 10, 8],
            "colsample_bytree": [0.8, 0.9, 1],
            "min_child_samples": [20],
            "min_child_weight": [1e-5],
            "subsample_freq": [10],
            "subsample": [0.8, 0.9, 1.0],
            "n_jobs": [10],
        }

        grid_search_func(train_data, train_label, model, params, self.save_path_philips)

        with open(self.save_path_philips, "rb") as f:
            info = pickle.load(f)

        model = LGBMClassifier(**info.get("best_params_"))
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
        model = LGBMClassifier(**info.get("best_params_"))

        for train_idx, test_idx in self.kfold.split(
            self.data_philips, self.label_philips
        ):
            # print(train_idx, test_idx)
            x_train, x_test = self.data.iloc[train_idx], self.data.iloc[test_idx]
            y_train, y_test = self.label.iloc[train_idx], self.label.iloc[test_idx]

            model.fit(x_train, y_train, verbose=False)

            fold_pred_train = model.score(x_train, y_train)
            fold_pred = model.predict(x_test)
            fold_pred_proba = model.predict_proba(x_test)[:, 1]

            fper, tper, threshold = roc_curve(
                self.label_philips.iloc[test_idx].values.ravel(), fold_pred_proba
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
        plt.title("ROC - LGBM")
        plt.legend(loc="lower right")
        plt.show()


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    cat = LightGBM(pca=True)
    # print("total")
    # cat.total_train_fold()
    print("total_test")
    cat.test_total()
    # print("siemens")
    # cat.train_siemens()
    print("siemens_test")
    cat.siemens()
    # print("philips")
    # cat.train_philips()
    print("philips_test")
    cat.philips()
    # lgbm.train()
    # lgbm.test()
    # lgbm.feature_importance()
