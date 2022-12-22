import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from data_load import load_total_data
from utils import *

from matplotlib.colors import ListedColormap
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFECV


class Voter:
    def __init__(self):
        self.total_data, self.total_label, self.philips_data, self.philips_label, self.siemens_data, self.siemens_label = load_total_data()

        self.kfold = StratifiedKFold(n_splits=10)

        # self.xgb = XGBClassifier(colsample_bytree=0.8, eval_metric="auc", gamma=0.5, learning_rate=0.01, max_depth=3, min_child_weight=10, n_estimators=200, random_state=42, subsample=0.6)
        self.xgb = XGBClassifier(eval_metric='auc')
        # self.cat = CatBoostClassifier(verbose=False, learning_rate=0.001, random_state=42, reg_lambda=1, subsample=0.5)
        self.cat = CatBoostClassifier(verbose=False)
        self.bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=1000, max_samples=200, bootstrap=True, n_jobs=10)
        # self.lr = LogisticRegression(C=1.7575106248547894, penalty='l1', solver='liblinear')
        self.lr = LogisticRegression()
        # self.knn = KNeighborsClassifier(n_neighbors=10)
        self.knn = KNeighborsClassifier()
        self.svc = SVC(probability=True)
        self.dt = DecisionTreeClassifier()
        self.ann = MLPClassifier()
        # self.ann = MLPClassifier(activation='tanh', alpha=0.001, solver='adam', hidden_layer_sizes=100, learning_rate_init=0.001, early_stopping=False, warm_start=False)
        self.nb = GaussianNB()
        self.rf = RandomForestClassifier()
        self.lgbm = LGBMClassifier()

        self.sgd = CalibratedClassifierCV(SGDClassifier(), cv=10)
        self.ada = AdaBoostClassifier()

        self.rf = RandomForestClassifier()

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.total_data, self.total_label, test_size=0.2, random_state=42, stratify=self.total_label)

    @staticmethod
    def plot_decision_boundary(clf, X, y, axes=None, alpha=0.5, contour=True):
        if axes is None:
            axes = [-1.5, 2.5, -1, 1.5]

        x1s = np.linspace(axes[0], axes[1], 100)
        x2s = np.linspace(axes[2], axes[3], 100)
        x1, x2 = np.meshgrid(x1s, x2s)

        X_new = np.c_[x1.ravel(), x2.ravel()]
        y_pred = clf.predict(X_new).reshape(x1.shape)

        custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
        plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)

        if contour:
            custom_cmap2 = ListedColormap(['#7d7d58', '#4c4c7f', '#507d50'])
            plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)

        plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", alpha=alpha)
        plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", alpha=alpha)
        plt.axis(axes)
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

    def train(self):
        # lgbm = LGBMClassifier(colsample_bytree=0.9, learning_rate=0.1, max_depth=4, min_child_samples=20, min_child_weight=1e-05, n_estimators=300, n_jobs=10, objective='binary', reg_alpha=10, subsample=0.8, subsample_freq=10)

        ensemble_voter = VotingClassifier(
            estimators=[
                ('SVC', self.svc),
                ('XGB', self.xgb),
                ('Cat', self.cat),
            ], voting='soft'
        )

        ensemble_voter2 = VotingClassifier(
            estimators=[
                ('SVC', self.svc),
                ('CAT', self.cat),
                ('LR', self.lr),
                ('SGD', self.sgd),
                ('VOT', ensemble_voter),
            ], voting='soft'
        )

        ensemble_voter3 = VotingClassifier(
            estimators=[
                # ('SVC', self.svc),
                # ('ANN', self.ann),
                ('VOT', ensemble_voter),
                ('VOT2', ensemble_voter2)
            ], voting='soft'
        )

        # clfs = [self.lr, self.knn, self.svc, self.ann, self.xgb, self.cat, self.rf, ensemble_voter]
        clfs = [self.svc, self.xgb, self.cat, self.lgbm, self.lr, self.knn, self.dt, self.ann, self.sgd, self.ada, self.rf, ensemble_voter, ensemble_voter2, ensemble_voter3]

        for clf in clfs:
            clf.fit(self.train_x, self.train_y)
            y_pred = clf.predict(self.test_x)
            pred_proba = clf.predict_proba(self.test_x)[:, 1]
            fper, tper, threshold = roc_curve(self.test_y, pred_proba)
            auc_score = auc(fper, tper)
            print(clf.__class__.__name__, '- ACC:', round(accuracy_score(self.test_y, y_pred), 2))
            print(clf.__class__.__name__, '- AUC:', round(auc_score, 2))
            # J = tper - fper
            # idx = np.argmax(J)
            # best_threshold = threshold[idx]
            # sens, spec = tper[idx], 1 - fper[idx]
            # print(
            #     "Best threshold = %.3f, Sensitivity = %.3f, Specificity = %.3f"
            #     % (best_threshold, sens, spec)
            # )

    def train_bagging(self):
        bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=1000, max_samples=100, bootstrap=True, n_jobs=10)
        bag_clf.fit(self.train_x, self.train_y)
        y_pred = bag_clf.predict(self.test_x)
        pred_proba = bag_clf.predict_proba(self.test_x)[:, 1]
        fper, tper, _ = roc_curve(self.test_y, pred_proba)
        auc_score = auc(fper, tper)
        print(bag_clf.__class__.__name__, accuracy_score(self.test_y, y_pred))
        print(bag_clf.__class__.__name__, auc_score)

        tree_clf = DecisionTreeClassifier(random_state=42)
        tree_clf.fit(self.train_x, self.train_y)
        y_pred_tree = tree_clf.predict(self.test_x)
        pred_proba = tree_clf.predict_proba(self.test_x)[:, 1]
        fper, tper, _ = roc_curve(self.test_y, pred_proba)
        auc_score = auc(fper, tper)
        print(tree_clf.__class__.__name__, accuracy_score(self.test_y, y_pred_tree))
        print(tree_clf.__class__.__name__, auc_score)

    @staticmethod
    def get_stacking_data(model, train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame, n_folds: int = 5):
        # kfold = StratifiedKFold(n_splits=n_folds)
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        train_fold_predict = np.zeros((train_x.shape[0], 1))
        test_predict = np.zeros((test_x.shape[0], n_folds))
        print(model.__class__.__name__, '시작')

        for idx, (train_idx, valid_idx) in enumerate(kfold.split(train_x, train_y)):
            train_data, valid_data = train_x.iloc[train_idx], train_x.iloc[valid_idx]
            train_label = train_y.iloc[train_idx]

            model.fit(train_data, train_label)

            train_fold_predict[valid_idx, :] = model.predict(valid_data).reshape(-1, 1)
            test_predict[:, idx] = model.predict(test_x)

        test_predict_mean = np.mean(test_predict, axis=1).reshape(-1, 1)

        return train_fold_predict, test_predict_mean

    @staticmethod
    def get_stacking_data_simple(model, train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame, test_y: pd.DataFrame):
        model.fit(train_x, train_y)

        pred = model.predict(test_x)

        print(model.__class__.__name__, '- ACC:', accuracy_score(test_y, pred))

        return pred

    @staticmethod
    def stacking_acc(model_cls, train_x, test_x, train_y, test_y, params, save_path):
        grid_search_func(train_x, train_y, model_cls, params, save_path)

        with open(save_path, "rb") as f:
            info = pickle.load(f)

        model = model_cls(**info.get("best_params_"))
        model.fit(train_x, train_y)

        pred = model.predict(test_x)
        pred_proba = model.predict_proba(test_x)[:, 1]

        fper, tper, threshold = roc_curve(test_y.values.ravel(), pred_proba)
        roc_auc = auc(fper, tper)
        accuracy = np.round(accuracy_score(test_y, pred), 4)

        print(f"정확도: {accuracy}")
        print(f"AUC: {roc_auc}")

        J = tper - fper
        idx = np.argmax(J)
        best_threshold = threshold[idx]
        sens, spec = tper[idx], 1 - fper[idx]
        print(
            "Best threshold = %.3f, Sensitivity = %.3f, Specificity = %.3f"
            % (best_threshold, sens, spec)
        )

    def train_stacking(self):
        sv_train, sv_test = self.get_stacking_data(self.svc, self.train_x, self.train_y, self.test_x)
        cat_train, cat_test = self.get_stacking_data(self.cat, self.train_x, self.train_y, self.test_x)
        lgbm_train, lgbm_test = self.get_stacking_data(self.lgbm, self.train_x, self.train_y, self.test_x)
        lr_train, lr_test = self.get_stacking_data(self.lr, self.train_x, self.train_y, self.test_x)
        knn_train, knn_test = self.get_stacking_data(self.knn, self.train_x, self.train_y, self.test_x)
        ann_train, ann_test = self.get_stacking_data(self.ann, self.train_x, self.train_y, self.test_x)
        sgd_train, sgd_test = self.get_stacking_data(self.sgd, self.train_x, self.train_y, self.test_x)
        xgb_train, xgb_test = self.get_stacking_data(self.xgb, self.train_x, self.train_y, self.test_x)
        dt_train, dt_test = self.get_stacking_data(self.dt, self.train_x, self.train_y, self.test_x)
        ada_train, ada_test = self.get_stacking_data(self.ada, self.train_x, self.train_y, self.test_x)

        x_train_total = np.concatenate((sv_train, cat_train, lgbm_train, lr_train, knn_train, ann_train, sgd_train, xgb_train), axis=1)
        x_test_total = np.concatenate((sv_test, cat_test, lgbm_test, lr_test, knn_test, ann_test, sgd_test, xgb_test), axis=1)

        x_train_total = pd.DataFrame(x_train_total, columns=['SVC', 'CAT', 'LGBM', 'LR', 'KNN', 'ANN', 'SGD', 'XGB'])
        x_test_total = pd.DataFrame(x_test_total, columns=['SVC', 'CAT', 'LGBM', 'LR', 'KNN', 'ANN', 'SGD', 'XGB'])

        print(f"원본: {self.train_x.shape}, {self.test_x.shape}")
        print(f"stack: {x_train_total.shape}, {x_test_total.shape}")

        test_x, valid_x, test_y, valid_y = train_test_split(x_test_total, self.test_y, test_size=0.5, random_state=42)

        # print("********** XGBoost **********")
        #
        # test_x, valid_x, test_y, valid_y = train_test_split(x_test_total, self.test_y, test_size=0.5, random_state=42)
        #
        # xgb = XGBClassifier(eval_metric="auc")
        # xgb.fit(x_train_total, self.train_y, eval_set=[(valid_x, valid_y)], early_stopping_rounds=100, verbose=False)
        # # xgb.fit(x_train_total, self.train_y, verbose=False)
        #
        # pred = xgb.predict(test_x)
        # # pred = xgb.predict(x_train_total)
        # pred_proba = xgb.predict_proba(test_x)[:, 1]
        # # pred_proba = xgb.predict_proba(x_train_total)[:, 1]
        #
        # fper, tper, threshold = roc_curve(test_y.values.ravel(), pred_proba)
        # # fper, tper, threshold = roc_curve(self.train_y.values.ravel(), pred_proba)
        # roc_auc = auc(fper, tper)
        #
        # accuracy = np.round(accuracy_score(test_y, pred), 4)
        # # accuracy = np.round(accuracy_score(self.train_y, pred), 4)
        #
        # print(f"정확도: {accuracy}")
        # print(f"AUC: {roc_auc}")

        print("********** Voter **********")

        xgb = VotingClassifier(
            estimators=[
                ('SVC', self.svc),
                ('XGB', self.xgb),
                ('Cat', self.cat),
            ], voting='soft'
        )

        xgb.fit(x_train_total, self.train_y)
        # xgb.fit(x_train_total, self.train_y, verbose=False)

        pred = xgb.predict(test_x)
        # pred = xgb.predict(x_train_total)
        pred_proba = xgb.predict_proba(test_x)[:, 1]
        # pred_proba = xgb.predict_proba(x_train_total)[:, 1]

        fper, tper, threshold = roc_curve(test_y.values.ravel(), pred_proba)
        # fper, tper, threshold = roc_curve(self.train_y.values.ravel(), pred_proba)
        roc_auc = auc(fper, tper)

        accuracy = np.round(accuracy_score(test_y, pred), 4)
        # accuracy = np.round(accuracy_score(self.train_y, pred), 4)

        print(f"정확도: {accuracy}")
        print(f"AUC: {roc_auc}")

    def train_stacking_simple(self):
        train_x, test_x, train_y, test_y = train_test_split(self.total_data, self.total_label, test_size=0.2, random_state=42, shuffle=True)

        svm_pred = self.get_stacking_data_simple(self.svc, train_x, train_y, test_x, test_y)
        xgb_pred = self.get_stacking_data_simple(self.xgb, train_x, train_y, test_x, test_y)
        cat_pred = self.get_stacking_data_simple(self.cat, train_x, train_y, test_x, test_y)

        new_data = np.array([svm_pred, xgb_pred, cat_pred])
        new_data = np.transpose(new_data)

        lgbm = LGBMClassifier()
        lgbm.fit(new_data, test_y)

        pred = lgbm.predict(new_data)
        pred_proba = lgbm.predict_proba(new_data)[:, 1]

        fper, tper, threshold = roc_curve(test_y.values.ravel(), pred_proba)
        roc_auc = auc(fper, tper)

        accuracy = np.round(accuracy_score(test_y, pred), 4)

        print(f"정확도: {accuracy}")
        print(f"AUC: {roc_auc}")



warnings.filterwarnings('ignore')


if __name__ == "__main__":
    voter = Voter()
    # voter.train()
    # voter.train_bagging()
    voter.train_stacking()
    # voter.train_stacking_simple()
    # voter.rfe()
    # voter.fs_train()
