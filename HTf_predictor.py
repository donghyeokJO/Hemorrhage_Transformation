import os
import glob
import re
import sys

import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt


def remove_ct_noise(img_arr):
    skull_pixel = 100
    black_pixel = -23

    img_arr = np.where(img_arr < black_pixel, black_pixel, img_arr)
    # img_arr = np.where(img_arr > skull_pixel, skull_pixel, img_arr)

    for i in range(img_arr.shape[0]):

        # Remove left
        for j in range(img_arr.shape[1]):
            for k in range(img_arr.shape[2] // 2):
                if img_arr[i][j][k] >= skull_pixel:
                    img_arr[i, j, :k] = black_pixel
                    break

        # Remove Right
        for j in range(img_arr.shape[1]):
            for k in reversed(range(img_arr.shape[2] // 2)):
                k = k + (img_arr.shape[2] // 2)
                if img_arr[i][j][k] >= skull_pixel:
                    img_arr[i, j, k + 1 :] = black_pixel
                    break

        # Remove Top
        for k in range(img_arr.shape[2]):
            for j in range(img_arr.shape[1] // 2):
                if img_arr[i][j][k] >= skull_pixel:
                    img_arr[i, :j, k] = black_pixel
                    break

        # Remove Right
        for k in range(img_arr.shape[2]):
            for r in reversed(range(img_arr.shape[1] // 2)):
                j = r + (img_arr.shape[1] // 2)
                if img_arr[i][j][k] >= skull_pixel:
                    img_arr[i, j + 1 :, k] = black_pixel
                    break

        img_arr[i][img_arr[i] >= skull_pixel] = black_pixel

        # Remove Top
        for j in range(img_arr.shape[1] // 2):
            # if img_arr[i][j] not np.any(black_pixel):
            # if np.all(img_arr[i][j] == black_pixel):
            if np.all(img_arr[i][j] == black_pixel):
                img_arr[i, :j] = black_pixel

        # Remove Bottom
        for j in reversed(range(img_arr.shape[1] // 2)):
            j = j + (img_arr.shape[1] // 2)
            if np.all(img_arr[i][j] == black_pixel):
                img_arr[i, j:] = black_pixel

        # Remove left
        for j in range(img_arr.shape[2] // 2):
            # if img_arr[i][j] not np.any(black_pixel):
            if np.all(img_arr[i, :, j] == black_pixel):
                img_arr[i, :, :j] = black_pixel

        # Remove right
        for r in reversed(range(img_arr.shape[1] // 2)):
            j = r + (img_arr.shape[1] // 2)
            if np.all(img_arr[i, :, j] == black_pixel):
                img_arr[i, :, j:] = black_pixel

    return img_arr


if __name__ == "__main__":

    dir = os.path.dirname(os.path.realpath("__file__"))
    data_dir = os.path.join(dir, "data")
    # data_dir = "data"
    dcm_info = os.path.join(dir, "data", "patient.xlsx")
    # dcm_info = "data/patient.xlsx"
    save_path = os.path.join(dir, "result_data/")
    # save_path = "result_data/"

    # total_img = np.empty((1,256,256,5))#np.array([])
    # total_seg = np.empty((1,256,256,2))#np.array([])
    total_list = []

    dcm_info = pd.read_excel(dcm_info)
    dcm_info = np.array(dcm_info)

    data_sets = glob.glob(data_dir + "/*/*")

    HUS = dict()
    for idx in range(len(dcm_info)):
        # if idx == 0:
        #     continue
        patient_num = str(dcm_info[idx, 1])
        # print(patient_num)
        patient_dir = [dir for dir in data_sets if patient_num in dir]
        patient_dir = patient_dir[0]
        a = patient_dir.split("\\")[-1]
        print(a)
        # if a != "200125286":
        #     continue

        HU = np.zeros(80)

        dcm_list = glob.glob(patient_dir + "/*.dcm")
        # print(dcm_list)
        # dcm_list = dcm_list[int(dcm_info[idx, 4]) : int(dcm_info[idx, 5])]

        result_df = pd.DataFrame()
        flag = False
        # for j, file in enumerate(reversed(dcm_list)):
        for j, file in enumerate(dcm_list):
            img = sitk.ReadImage(file)
            img_arr = sitk.GetArrayFromImage(img)
            img_arr = remove_ct_noise(img_arr)
            # print(len(np.unique(img_arr)))
            if len(np.unique(img_arr)) > 120 and not flag:
                continue

            if len(np.unique(img_arr)) <= 120 and not flag:
                print(f"Start index: {j+1}")
                flag = True
                pass

            if len(np.unique(img_arr)) <= 120 and flag:
                pass

            if len(np.unique(img_arr)) > 120 and flag:
                print(f"Last index: {j}")
                break
            rm_img = sitk.GetImageFromArray(img_arr)

            writer = sitk.ImageFileWriter()
            # writer.SetFileName(f"removed Image{len(dcm_list)-j}.dcm")
            writer.SetFileName(f"removed Image{j + 1}.dcm")
            writer.Execute(rm_img)
            # break
            for i in range(0, 80):
                HU[i] = HU[i] + img_arr[0].flatten().tolist().count(i)

        # dcm_info[idx, 7:] = HU
        HUS[a] = HU
        # if idx == 1:
        #     break

    df = pd.DataFrame(HUS).T
    df.to_excel(os.path.join(save_path, "HU_list.xlsx"), index=False)
