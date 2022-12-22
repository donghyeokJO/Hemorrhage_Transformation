import copy
import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from math import sqrt
from scipy.stats import sem, t, norm
from scipy.special import ndtri
from sklearn.metrics import roc_curve, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterGrid


def predication_func(x: pd.DataFrame, y: pd.DataFrame, classify, param):
    kfold = StratifiedKFold(n_splits=10)

    pred_list, real_list, prob_list = [], [], []
    # print(param)
    for train_index, test_index in kfold.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf = classify(**param)
        try:
            clf.fit(x_train, y_train, verbose=False)
            # clf.fit(x_train, y_train)
        except:
            clf.fit(x_train, y_train)

        pred_prob = clf.predict_proba(x_test)[:, 1]
        pred = clf.predict(x_test)

        pred_list += list(pred)
        real_list += list(y_test.values.ravel())
        prob_list += list(pred_prob)

    pred_list = np.array(pred_list).reshape(-1)
    real_list = np.array(real_list).reshape(-1)
    prob_list = np.array(prob_list).reshape(-1)
    return pred_list, real_list, prob_list


def predication_func_train_test_split(
    train_data: pd.DataFrame,
    train_label: pd.DataFrame,
    test_data: pd.DataFrame,
    test_label: pd.DataFrame,
    classify,
    param,
):
    pred_list, real_list, prob_list, pred_list_test, prob_list_test, real_list_test = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    clf = classify(**param)

    try:
        clf.fit(train_data, train_label, verbose=False)
    except:
        clf.fit(train_data, train_label)

    pred_list += list(clf.predict(train_data))
    real_list += list(train_label.values.ravel())
    prob_list += list(clf.predict_proba(train_data)[:, 1])

    pred_list_test += list(clf.predict(test_data))
    real_list_test += list(test_label.values.ravel())
    prob_list_test += list(clf.predict_proba(test_data)[:, 1])

    pred_list = np.array(pred_list).reshape(-1)
    real_list = np.array(real_list).reshape(-1)
    prob_list = np.array(prob_list).reshape(-1)

    pred_list_test = np.array(pred_list_test).reshape(-1)
    real_list_test = np.array(real_list_test).reshape(-1)
    prod_list_test = np.array(prob_list_test).reshape(-1)

    return (
        pred_list,
        real_list,
        prob_list,
        pred_list_test,
        real_list_test,
        prod_list_test,
    )


def _proportion_confidence_interval(r, n, z):
    A = 2 * r + z**2
    B = z * sqrt(z**2 + 4 * r * (1 - r / n))
    C = 2 * (n + z**2)

    return (A - B) / C, (A + B) / C


def sensitivity_and_specificity_with_confidence_intervals(TP, FP, FN, TN, alpha=0.95):
    z = -ndtri((1.0 - alpha) / 2)

    sensitivity_point_estimate = TP / (TP + FN)
    sensitivity_confidence_interval = _proportion_confidence_interval(TP, TP + FN, z)

    specificity_point_estimate = TN / (TN + FP)
    specificity_confidence_interval = _proportion_confidence_interval(TN, TN + FP, z)

    return (
        sensitivity_point_estimate,
        specificity_point_estimate,
        sensitivity_confidence_interval,
        specificity_confidence_interval,
    )


def ppv_and_npv_with_confidence_intervals(TP, FP, FN, TN, alpha=0.95):
    z = -ndtri((1.0 - alpha) / 2)

    ppv_estimate = TP / (TP + FP)
    ppv_confidence_interval = _proportion_confidence_interval(TP, TP + FP, z)

    npv_estimate = TN / (TN + FN)
    npv_confidence_interval = _proportion_confidence_interval(TN, TN + FN, z)

    return ppv_estimate, npv_estimate, ppv_confidence_interval, npv_confidence_interval


def find_optimal_cutoff(real, pred):
    fpr, tpr, thresholds = roc_curve(real, pred, pos_label=1)
    ix = np.argmin((1 - tpr) ** 2 + (1 - (1 - fpr)) ** 2)

    return ix


def roc_curve_func(pred, real, prob, save_file):
    ix = find_optimal_cutoff(real, prob)
    fpr, tpr, thresholds = roc_curve(real, prob, pos_label=1)

    plt.clf()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.scatter(fpr[ix], tpr[ix], marker="o", color="black", label="Best")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")

    plt.savefig(save_file, format="eps", rasterized=True)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2.0, n - 1)

    return m, h


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0

    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j

    T2 = np.empty(N, dtype=np.float)
    T2[J] = T + 1

    return T2


def compute_midrank_weight(x, sample_weight):
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0

    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j

    T2 = np.empty(N, dtype=np.float)
    T2[J] = T

    return T2


def fastDeLong_no_weights(predications_sorted_transposed, label_1_count):
    m = label_1_count
    n = predications_sorted_transposed.shape[1] - m

    positive_examples = predications_sorted_transposed[:, :m]
    negative_examples = predications_sorted_transposed[:, m:]

    k = predications_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predications_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01)
    sy = np.cov(v10)

    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_weights(pred_sorted_transposed, label_1_count, sample_weight):
    m = label_1_count
    n = pred_sorted_transposed.shape[1] - m

    positive_examples = pred_sorted_transposed[:, :m]
    negative_examples = pred_sorted_transposed[:, m:]

    k = pred_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)

    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(pred_sorted_transposed[r, :], sample_weight)

    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()

    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()

    aucs = (sample_weight[:m] * (tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / total_positive_weights

    sx = np.cov(v01)
    sy = np.cov(v10)

    delongcov = sx / m + sy / n

    return aucs, delongcov


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(
            predictions_sorted_transposed, label_1_count, sample_weight
        )


def calc_pvalue(aucs, sigma):
    l_aux = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l_aux, sigma)), l_aux.T)

    return np.log10(2) + norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()

    label_1_count = int(ground_truth.sum())

    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    ground_truth_stats = compute_ground_truth_statistics(ground_truth, sample_weight)
    order, label_1_count, ordered_sample_weight = ground_truth_stats

    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(
        predictions_sorted_transposed, label_1_count, ordered_sample_weight
    )

    assert_msg = "There is a bug in the code."

    assert len(aucs) == 1, assert_msg
    return aucs[0], delongcov


def delong_roc_test(ground_truth, pred_one, pred_two, sample_weight=None):
    order, label_1_count, _ = compute_ground_truth_statistics(
        ground_truth, sample_weight
    )

    predictions_sorted_transposed = np.vstack((pred_one, pred_two))[:, order]

    aucs, delongcov = fastDeLong(
        predictions_sorted_transposed, label_1_count, sample_weight
    )

    return calc_pvalue(aucs, delongcov)


def auc_ci_Delong(y_true, y_scores, alpha=0.95):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    auc, auc_var = delong_roc_variance(y_true, y_scores)

    auc_std = np.sqrt(auc_var)

    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    lower_upper_ci = norm.ppf(lower_upper_q, loc=auc, scale=auc_std)

    lower_upper_ci[lower_upper_ci > 1] = 1

    return auc, auc_var, lower_upper_ci


def grid_search_func(x, y, classify, param_grid, save_path):
    best_score = None
    best_parameter = None

    try:
        with open(save_path, "rb") as f:
            data = pickle.load(f)
            best_score = data["best_score_"]
    except:
        pass

    total_size = len(ParameterGrid(param_grid))

    for i, param_ in enumerate(ParameterGrid(param_grid)):
        param_ = copy.deepcopy(param_)
        pred_list, real_list, prob_list = predication_func(x, y, classify, param_)

        auc_result = auc_ci_Delong(real_list, prob_list)

        if best_score is None:
            best_score = auc_result[0]
            best_parameter = param_
            continue

        if best_score <= auc_result[0]:
            best_score = auc_result[0]
            best_parameter = param_

        print(
            "{0:03d} / {1:03d} : {2} {3}".format(
                i, total_size, best_parameter, best_score
            )
        )

        with open(save_path, "wb") as f:
            pickle.dump({"best_score_": best_score, "best_params_": best_parameter}, f)