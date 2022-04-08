import os
import glob
import pydicom

import numpy as np
import pandas as pd

from get_hounsfield import remove_ct_noise


def load_data():
    dir = os.path.dirname(os.path.realpath("__file__"))
    data_dir = os.path.join(dir, "data")
    dcm_info = os.path.join(dir, "data", "patient.xlsx")

    dcm_info = pd.read_excel(dcm_info)
    dcm_info = np.array(dcm_info)

    data_sets = glob.glob(data_dir + "/*/*")

    data_list = []
    label_list = []

    for idx in range(len(dcm_info[:40])):
        patient_num = str(dcm_info[idx, 1])
        patient_dir = [dir for dir in data_sets if patient_num in dir]
        patient_dir = patient_dir[0]

        dcm_list = glob.glob(patient_dir + "/*.dcm")

        data_list_temp = []
        label_list_temp = []

        eye_flag = True

        for i, file in enumerate(dcm_list):
            medical_image = pydicom.dcmread(file)
            masked_image = remove_ct_noise(medical_image)
            masked_image.reshape((512, 512, 1))

            if "no_eye" in file:
                eye_flag = False
            label = 1 if eye_flag else 0

            data_list_temp.append(masked_image)
            label_list_temp.append(label)

        data_list += data_list_temp
        label_list += label_list_temp

    return np.stack(data_list), np.stack(label_list)


if __name__ == "__main__":
    data_list, label_list = load_data()
    print(data_list)
    print(label_list)
