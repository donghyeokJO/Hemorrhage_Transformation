import os
import glob
import pydicom
import logging

import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage import morphology
from scipy import ndimage
from scipy.stats import kurtosis, skew


def transform_to_hu(medical_image, image):
    # image = medical_image.pixel_array

    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept

    return hu_image


def remove_ct_noise(medical_image: pydicom.FileDataset, image) -> np.ndarray:
    hu_image = transform_to_hu(medical_image, image)
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


if __name__ == "__main__":
    # multi producer
    # producers = ["gms", "philips", "toshiba"]
    producers = ["toshiba"]
    dir = os.path.dirname(os.path.realpath("__file__"))

    # data_dir = os.path.join(dir, "predictions")
    save_path = os.path.join(dir, "masks")
    hu_path = os.path.join(dir, "result_data")

    for prod in producers:
        data_dir = os.path.join(dir, f"predictions_{prod}")

        data_sets = glob.glob(data_dir + "/*/*")

        # dicom_dir = os.path.join(dir, "data/Dicom_files")
        dicom_dir = os.path.join(dir, f"data/Dicom_New_{prod}")

        dicom_dir_list = os.listdir(dicom_dir)
        patients_masks = os.listdir(data_dir)

        HUS = []
        skewness_ = []
        kurtosis_ = []
        hosp_ids = []

        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)

        for patient in tqdm(dicom_dir_list):
            try:
                patient_mask = f"{patient}.nii.gz"

                if patient_mask not in patients_masks:
                    print("변환 파일 없음: " + patient)
                    continue

                # hosp_ids.append(patient)

                mask_img = nib.load(os.path.join(data_dir, patient_mask))
                img_arr = mask_img.get_fdata()
                mask_arr = []

                for i in range(img_arr.shape[-1]):
                    im_arr = img_arr[:, :, i]
                    im_arr = np.rot90(im_arr, axes=(0, 1))
                    im_arr = np.rot90(im_arr, axes=(0, 1))
                    im_arr = np.rot90(im_arr, axes=(0, 1))
                    mask_arr.append(im_arr)

                dicom_path = os.path.join(dicom_dir, patient)
                dicom_files = os.listdir(dicom_path)

                HU = np.zeros(80)

                for idx, file in enumerate(dicom_files):
                    a = pydicom.dcmread(os.path.join(dicom_path, file))
                    img = a.pixel_array

                    mask_ = mask_arr[idx]

                    img = mask_ * img
                    masked_img = remove_ct_noise(a, img)

                    # plt.imshow(masked_img, cmap="gray")
                    # plt.show()

                    for i in range(0, 80):
                        HU[i] += masked_img.flatten().tolist().count(i)

                HUS.append(HU)
                hosp_ids.append(patient)

            except Exception as err:
                print(patient)
                logger.error(repr(err))
                pass

        for hu in HUS:
            skewness_.append(skew(hu))
            kurtosis_.append(kurtosis(hu, fisher=True))

        results = {
            "hosp_id": hosp_ids,
            "skewness": skewness_,
            "kurtosis": kurtosis_,
        }

        for i in range(0, 80):
            results["HU" + str(i)] = list(map(lambda h: h[i], HUS))

        df = pd.DataFrame(results)
        df.to_excel(os.path.join(hu_path, f"HU_{prod}.xlsx"), header=True, index=None)
