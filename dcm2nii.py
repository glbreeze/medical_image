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
folder = '/scratch/yx2432/dataset/CT'
ImageFolder = 'Image'
MaskFolder = 'Mask'
TargetFolder = 'ct'

# ============= convert .dcm file to .nii file =============

if __name__ == '__main__':
    nii_folder = os.path.join('../dataset/CT', TargetFolder)
    if not os.path.exists(nii_folder):
        os.makedirs(nii_folder) 
            
    for subfolder in os.listdir(os.path.join(folder, ImageFolder)):
        if subfolder.startswith('.'):   # skip undesired folders
            continue
        if os.path.exists(os.path.join(nii_folder, subfolder + '.nii.gz')): 
            continue

        ct_dir = os.path.join(folder, ImageFolder, subfolder)

        reader = sitk.ImageSeriesReader()
        ct_names = reader.GetGDCMSeriesFileNames(ct_dir)
        reader.SetFileNames(ct_names)
        image = reader.Execute()
        fixed_image = sitk.Cast(image, sitk.sitkFloat32) # convert image to sitk.sitkFloat32

        # store the image as nii file
        sitk.WriteImage(fixed_image, os.path.join(nii_folder, subfolder + '.nii.gz'))
        print('--finished converting image {}'.format(subfolder))