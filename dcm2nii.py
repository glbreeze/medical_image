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
ImageFolder = 'image'
MaskFolder = 'mask'
TargetFolder = 'ct'

# ============= convert .dcm file to .nii file =============

if __name__ == '__main__':
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