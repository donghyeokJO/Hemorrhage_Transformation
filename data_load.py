import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind


def load_excel_data(data_path: str = "data/patient.xlsx") -> pd.DataFrame:
    dataset = pd.read_excel(data_path)
    dataset = dataset.astype({"hosp_id": "str"})
    dataset = dataset.set_index(["hosp_id"])

    drop_columns = [
        "manufacturer",
        "Tube current",
        "Tube voltage",
        "dis_mrs",
        "iv_start",
        "ia_start",
        "ia_end",
        "type_sub",
        "type",
        "reg_num",
    ]

    dataset.drop(drop_columns, axis=1, inplace=True)

    return dataset


def load_hu_data(data_path: str = "result_data/HU_list.xlsx") -> pd.DataFrame:
    dataset = pd.read_excel(data_path)
    dataset = dataset.astype({"hosp_id": "str"})
    dataset = dataset.set_index(["hosp_id"])

    return dataset


def pca_data(new_dataset: pd.DataFrame) -> pd.DataFrame:
    hu_columns = ["HU" + str(x) for x in range(0, 80)]
    hu_df = new_dataset.loc[:, new_dataset.columns.isin(hu_columns)]

    # pca = PCA()
    # pca.fit(hu_df)
    # exp_var_ = np.cumsum(pca.explained_variance_ratio_)
    # print(f"HU: {exp_var_}")

    pca = PCA(n_components=5)
    hu_principal = pca.fit_transform(hu_df)

    for i in range(0, 5):
        new_dataset.insert(i, "HU_" + str(i), hu_principal[:, i])

    new_dataset.drop(hu_columns, axis=1, inplace=True)

    # personal_info = ["male", ""]
    return new_dataset


def pca_data_segmented(new_dataset: pd.DataFrame) -> pd.DataFrame:
    hu_col1 = ["HU" + str(x) for x in range(0, 16)]
    hu_col2 = ["HU" + str(x) for x in range(16, 40)]
    hu_col3 = ["HU" + str(x) for x in range(40, 80)]

    hu_df1 = new_dataset.loc[:, new_dataset.columns.isin(hu_col1)]
    hu_df2 = new_dataset.loc[:, new_dataset.columns.isin(hu_col2)]
    hu_df3 = new_dataset.loc[:, new_dataset.columns.isin(hu_col3)]

    pca1 = PCA(n_components=1)
    pca2 = PCA(n_components=2)
    pca3 = PCA(n_components=2)

    hu_df1_principal = pca1.fit_transform(hu_df1)
    hu_df2_principal = pca2.fit_transform(hu_df2)
    hu_df3_principal = pca3.fit_transform(hu_df3)

    new_dataset.insert(0, "HU_SEG1", hu_df1_principal)

    new_dataset.insert(1, "HU_SEG2_1", hu_df2_principal[:, 0])
    new_dataset.insert(2, "HU_SEG2_2", hu_df2_principal[:, 1])
    new_dataset.insert(3, "HU_SEG3_1", hu_df3_principal[:, 0])
    new_dataset.insert(4, "HU_SEG3_2", hu_df3_principal[:, 1])

    new_dataset.drop(hu_col1, axis=1, inplace=True)
    new_dataset.drop(hu_col2, axis=1, inplace=True)
    new_dataset.drop(hu_col3, axis=1, inplace=True)

    return new_dataset


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data1 = load_excel_data()
    data2 = load_hu_data()

    new_dataset = pd.concat([data1, data2], axis=1)
    labels = pd.DataFrame(new_dataset.loc[:, ["HTf"]])
    new_dataset.drop(["HTf"], axis=1, inplace=True)
    columns = new_dataset.columns

    scaler = MinMaxScaler()

    new_dataset = pd.DataFrame(
        scaler.fit_transform(new_dataset),
        columns=columns,
        index=new_dataset.index,
    )

    new_dataset = pca_data_segmented(new_dataset)
    new_dataset = new_dataset.fillna(0)
    return new_dataset, labels


def hu_distribution():
    data1 = load_excel_data()
    data2 = load_hu_data()

    new_dataset = pd.concat([data1, data2], axis=1)
    scaler = MinMaxScaler()
    total_dataset = pd.DataFrame(
        scaler.fit_transform(new_dataset),
        columns=new_dataset.columns,
    )

    HTf_condition = total_dataset.loc[:, "HTf"] == 1
    HTf_no_condition = total_dataset.loc[:, "HTf"] == 0

    HTf_yes = total_dataset.loc[HTf_condition]
    HTf_no = total_dataset.loc[HTf_no_condition]

    hu_columns = ["HU" + str(x) for x in range(0, 80)]

    HTf_yes = HTf_yes.loc[:, HTf_yes.columns.isin(hu_columns)]
    HTf_no = HTf_no.loc[:, HTf_no.columns.isin(hu_columns)]
    print(HTf_yes.shape)
    # fig = plt.figure(figsize=[12, 12])
    # ax = fig.add_subplot(111, aspect="equal")

    HTf_yes_mean = HTf_yes.mean(axis=0)
    HTf_no_mean = HTf_no.mean(axis=0)

    HTf_yes_std = HTf_yes.std()
    HTf_no_std = HTf_no.std()

    HTf_yes_reliability = []
    HTf_no_reliability = []

    for i in HTf_yes_std:
        HTf_yes_reliability.append(1.96 * i / math.sqrt(HTf_yes.shape[0]))

    for i in HTf_no_std:
        HTf_no_reliability.append(1.96 * i / math.sqrt(HTf_no.shape[0]))

    upper_ci_HTf_yes = []
    upper_ci_HTf_no = []
    lower_ci_HTf_yes = []
    lower_ci_HTf_no = []

    for i in range(len(HTf_yes_mean)):
        upper_ci_HTf_yes.append(HTf_yes_mean[i] + HTf_yes_reliability[i])
        lower_ci_HTf_yes.append(HTf_yes_mean[i] - HTf_yes_reliability[i])

    for i in range(len(HTf_no_mean)):
        upper_ci_HTf_no.append(HTf_no_mean[i] + HTf_no_reliability[i])
        lower_ci_HTf_no.append(HTf_no_mean[i] - HTf_no_reliability[i])

    fig, ax = plt.subplots()
    # fig.set_size_inches(15, 15)
    x = [x for x in range(0, 80)]
    ax.fill_between(
        x,
        upper_ci_HTf_yes,
        lower_ci_HTf_yes,
        facecolor="red",
        interpolate=True,
        alpha=0.2,
    )

    ax.plot(x, HTf_yes_mean, "r", label="HTf")
    ax.set_xlabel("Hounsfield Unit")
    ax.set_ylabel("Mean value")
    ax.set_title("Hounsfield Unit distribution (CI 95%)")

    ax.fill_between(
        x,
        upper_ci_HTf_no,
        lower_ci_HTf_no,
        facecolor="green",
        interpolate=True,
        alpha=0.2,
    )

    ax.plot(x, HTf_no_mean, "g", label="no HTF")
    ax.legend(loc="upper right")
    plt.show()


def t_test():
    data1 = load_excel_data()
    data2 = load_hu_data()

    new_dataset = pd.concat([data1, data2], axis=1)
    scaler = MinMaxScaler()

    total_dataset = pd.DataFrame(
        scaler.fit_transform(new_dataset),
        columns=new_dataset.columns,
    )

    HTf_condition = total_dataset.loc[:, "HTf"] == 1
    HTf_no_condition = total_dataset.loc[:, "HTf"] == 0

    HTf_yes = total_dataset.loc[HTf_condition]
    HTf_no = total_dataset.loc[HTf_no_condition]

    hu_columns = ["HU" + str(x) for x in range(0, 80)]

    HTf_yes = HTf_yes.loc[:, HTf_yes.columns.isin(hu_columns)]
    HTf_no = HTf_no.loc[:, HTf_no.columns.isin(hu_columns)]

    for i in range(0, 80):
        if (
            ttest_ind(HTf_yes.loc[:, "HU" + str(i)], HTf_no.loc[:, "HU" + str(i)])[1]
            < 0.05
        ):
            print(
                f"t-test result HU{i}: {ttest_ind(HTf_yes.loc[:,'HU'+str(i)], HTf_no.loc[:,'HU'+str(i)])[1]}"
            )
        else:
            continue


if __name__ == "__main__":
    # t_test()
    # hu_distribution()
    data, label = load_data()
    print(np.where(label == np.inf))
    print(np.where(np.isnan(data)))
    # print(np.all(np.isfinite(data)))
    # print(label)
