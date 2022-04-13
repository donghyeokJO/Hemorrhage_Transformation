import pydicom
import os
import glob
import keras
import pickle

import numpy as np
import pandas as pd

from skimage import morphology
from scipy import ndimage
from scipy.stats import kurtosis, skew


def transform_to_hu(medical_image: pydicom.FileDataset) -> np.ndarray:
    image = medical_image.pixel_array

    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept

    # hu_image = apply_modality_lut(hu_image, medical_image)
    # hu_image = apply_voi_lut(hu_image, medical_image)

    return hu_image


def window_image(
    hu_image: np.ndarray, window_center: int, window_width: int
) -> np.ndarray:
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    window_img = hu_image.copy()

    window_img[window_img < img_min] = img_min
    window_img[window_img > img_max] = img_max

    return window_img


def remove_ct_noise(medical_image: pydicom.FileDataset) -> np.ndarray:
    hu_image = transform_to_hu(medical_image)
    # window_img = window_image(hu_image, 40, 80)
    window_img = hu_image
    segmentation = morphology.dilation(window_img, np.ones((5, 5)))
    labels, label_nb = ndimage.label(segmentation)

    label_count = np.bincount(labels.ravel().astype(int))

    # background
    label_count[0] = 0

    mask = labels == label_count.argmax()

    mask = morphology.dilation(mask, np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))

    masked_image = mask * window_img

    return masked_image


def get_eye_detect(model, data):
    result = model.predict(data)

    return [1 if x[0] > 0.5 else 0 for x in result]


if __name__ == "__main__":
    dir = os.path.dirname(os.path.realpath("__file__"))
    data_dir = os.path.join(dir, "data")
    dcm_info = os.path.join(dir, "data", "patient.xlsx")
    save_path = os.path.join(dir, "result_data/")

    total_list = []

    dcm_info = pd.read_excel(dcm_info)
    dcm_info = np.array(dcm_info)

    data_sets = glob.glob(data_dir + "/*/*")

    model = keras.models.load_model("eye_detector.save")
    model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["acc"])

    try:
        with open("hu.txt", "rb") as f:
            HUS = pickle.load(f).get("HUS")
        #     HUS = HUS[:-2]
        #
        # with open("hu.txt", "wb") as f:
        #     pickle.dump({"HUS": HUS}, f)
    except:
        HUS = []

    hosp_ids = []
    skewness_ = []
    kurtosis_ = []
    # mu = []

    file_flag = False
    for idx in range(len(dcm_info)):
        patient_num = str(dcm_info[idx, 1])
        patient_dir = [dir for dir in data_sets if patient_num in dir]
        patient_dir = patient_dir[0]

        a = patient_dir.split("\\")[-1]
        hosp_ids.append(a)

        if a == "000006247":
            file_flag = True

        if not file_flag:
            continue

        print(f"hosp_id: {a}")

        HU = np.zeros(80)
        dcm_list = glob.glob(patient_dir + "/*.dcm")

        dcms = []

        for j, file in enumerate(dcm_list):
            medical_image = pydicom.dcmread(file)
            masked_image = remove_ct_noise(medical_image)
            masked_image = masked_image.reshape((512, 512, 1))
            # print(np.unique(masked_image))
            dcms.append(masked_image)

        eye_result = get_eye_detect(model, data=np.stack(dcms))

        flag = False
        # for k, img in enumerate(reversed(dcms)):
        for k, img in enumerate(dcms):
            # if eye_result[len(dcms) - k - 1] == 1:
            #     print(f"break at {len(dcms) - k}")
            #     break

            if eye_result[k] == 0 and not flag:
                print(f"starts at {k+1}")
                flag = True

            if not flag:
                continue

            for i in range(0, 80):
                HU[i] += img.flatten().tolist().count(i)

        HUS.append(HU)
        with open("hu.txt", "wb") as f:
            pickle.dump({"HUS": HUS}, f)

    for hu in HUS:
        skewness_.append(skew(hu))
        kurtosis_.append(kurtosis(hu, fisher=True))
        # mu.append(np.mean(hu))

    results = {
        "hosp_id": hosp_ids,
        "skewness": skewness_,
        "kurtosis": kurtosis_,
        # "Mu": mu,
    }

    for i in range(0, 80):
        results["HU" + str(i)] = list(map(lambda h: h[i], HUS))

    # print(results)
    df = pd.DataFrame(results)
    df.to_excel(os.path.join(save_path, "HU_list.xlsx"), header=True, index=None)
