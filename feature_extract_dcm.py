"""
This file try to extract feature from .dcm files directly. Did not work out well.
"""

import pdb
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import pydicom
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import radiomics
from radiomics import featureextractor, imageoperations

# Load CT scan and segmentation mask
folder = '../dataset/CT/'
ImageFolder = 'image1'
MaskFolder = 'mask1'
TargetFolder = 'ct'

params = {
    "binWidth": 25,
    "interpolator": "sitkLinear",
    "label": 1,
    "normalize": True,
    "removeOutliers": True,
    "verbose": True,
    "geometryTolerance": 1e-6,
    "sigma":[1,2,3],
    "resampledPixelSpacing":[1,1,1]
}

# ============= convert .dcm file to .nii file =============

for subfolder in os.listdir(os.path.join(folder, ImageFolder)):
    if subfolder.startswith('.'):   # skip undesired folders
        continue

    ct_dir = os.path.join(folder, ImageFolder, subfolder)
    nii_folder = os.path.join(folder, TargetFolder)
    if not os.path.exists(nii_folder):
        os.makedirs(nii_folder)

    reader = sitk.ImageSeriesReader()
    ct_names = reader.GetGDCMSeriesFileNames(ct_dir)
    reader.SetFileNames(ct_names)
    image = reader.Execute()
    fixed_image = sitk.Cast(image, sitk.sitkFloat32) # convert image to sitk.sitkFloat32

    # store the image as nii file
    sitk.WriteImage(fixed_image, os.path.join(nii_folder, subfolder + '.nii.gz'))

# Load CT scan and segmentation mask
folder = '../dataset/CT/'
image_path = os.path.join(folder, 'Image1', '0800428919')
mask_path = os.path.join(folder, 'Mask1', '0800428919.nii.gz')

# Read DICOM files
def get_ct_img(image_path):
    dcm_slices = [pydicom.dcmread(os.path.join(image_path, file_name))
                            for file_name in sorted(os.listdir(image_path)) if file_name.endswith('.dcm')]

    dcm_slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    slices = [x.pixel_array[None, ...] for x in dcm_slices]
    pixel_data = np.concatenate(slices, axis=0)

    # Get image dimensions and voxel spacing
    width, height = int(dcm_slices[0].Rows), int(dcm_slices[0].Columns)
    slice_thickness = float(dcm_slices[0].SliceThickness)
    spacing = (float(dcm_slices[0].PixelSpacing[0]), float(dcm_slices[0].PixelSpacing[1]), slice_thickness)

    image = sitk.GetImageFromArray(pixel_data, isVector=False)
    image.SetSpacing(spacing)
    image.SetOrigin(dcm_slices[0].ImagePositionPatient)

    return image


# ==== read masks
image = get_ct_img(image_path)
mask = sitk.ReadImage(mask_path)
corrected_mask = imageoperations._correctMask(image, mask)

extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(**params)
features = extractor.execute(image, corrected_mask)

for key, value in features.items():
    print(key, ":", value)

