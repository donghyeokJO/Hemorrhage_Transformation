import pydicom
import cv2
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage import morphology
from scipy import ndimage


def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept

    return hu_image


def initial_mask(image, low: int = -40, high: int = 160):
    mask_image = image.copy()
    mask_image[mask_image < low] = 0
    mask_image[mask_image > high] = 0
    mask_image[low <= mask_image] = 1
    mask_image[high >= mask_image] = 1

    return mask_image * image
    # return mask_image


def discard_mask(mask_image):
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    threshold = 7
    cranial_threshold = 300


def remove_noise(image):
    segmentation = morphology.dilation(image, np.ones((5, 5)))
    # print(segmentation)
    labels, label_nb = ndimage.label(segmentation)
    label_count = np.bincount(labels.ravel().astype(int))
    label_count[0] = 0

    mask = labels == label_count.argmax()
    mask = morphology.dilation(mask, np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))

    masked_image = mask * image

    return masked_image


dicom_path = "data/Dicom_files/000006247"
file_list = os.listdir(dicom_path)

for file in file_list[:]:
    a = pydicom.dcmread(dicom_path + "/" + file)
    image = a.pixel_array

    hu_image = transform_to_hu(a, image)
    mask = initial_mask(hu_image)
    masked_image = remove_noise(mask)
    plt.imshow(mask, cmap="gray")
    # plt.imshow(masked_image, cmap="gray")
    plt.show()
