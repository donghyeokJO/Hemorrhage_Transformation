import dicom2nifti
import os
import glob

import dicom2nifti.settings as settings
import pandas as pd
import numpy as np
import nibabel as nib

from tqdm import tqdm

if __name__ == "__main__":
    dir = os.path.dirname(os.path.realpath("__file__"))
    data_dir = os.path.join(dir, "data")
    # dcm_info = os.path.join(dir, "data", "patient.xlsx")
    # dcm_info = os.path.join(data_dir, "patient_1024.xlsx")
    # save_path = os.path.join(dir, "nifti/")
    save_path = os.path.join(dir, "nifti_1024/")

    # dcm_info = pd.read_excel(dcm_info)
    # dcm_info = np.array(dcm_info)

    data_sets = glob.glob(data_dir + "/*/*")

    hosp_ids = []

    dicom_dir = os.path.join(data_dir, "Dicom_1024")
    patient_folders = os.listdir(dicom_dir)

    settings.disable_validate_orthogonal()
    settings.disable_validate_slice_increment()
    settings.enable_pydicom_read_force()

    for patient in tqdm(patient_folders[:]):
        try:
            file_save_path = os.path.join(save_path, patient)
            if not os.path.exists(file_save_path):
                os.makedirs(file_save_path)

            dicom2nifti.convert_directory(
                os.path.join(dicom_dir, patient),
                os.path.join(save_path, file_save_path),
            )

            saved_file = os.listdir(os.path.join(save_path, file_save_path))[0]

            nifti_img = nib.load(os.path.join(save_path, file_save_path, saved_file))

            new_img = np.flip(nifti_img.get_fdata(), axis=1)
            new_nifti = nib.Nifti1Image(new_img.astype(float), nifti_img.affine)
            nib.save(new_nifti, os.path.join(save_path, f"{patient}.nii.gz"))
        except Exception as err:
            print(patient)

            print(repr(err))
            pass
        # break
