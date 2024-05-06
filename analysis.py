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




import pdb
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
import radiomics
from radiomics import featureextractor, imageoperations


def resample_image(image, new_spacing):
    # Get the current spacing
    current_spacing = image.GetSpacing()

    # Calculate the resampling factor
    resampling_factor = [cs / ns for cs, ns in zip(current_spacing, new_spacing)]

    # Calculate the new size based on the resampling factor
    new_size = [int(image.GetSize()[i] * resampling_factor[i]) for i in range(image.GetDimension())]

    # Create the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)  # You can change the interpolator as needed

    # Resample the image
    resampled_image = resampler.Execute(image)

    return resampled_image


def get_info(image):
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    size = image.GetSize()

    print("Origin:", origin)
    print("Spacing:", spacing)
    print("Direction:", direction)
    print("Size:", size)


def extract_features(scan, mask):
    # Initialize PyRadiomics feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()

    # Disable PyRadiomics logging (optional)
    radiomics.logger.setLevel(radiomics.logging.ERROR)

    feature_dict = extractor.execute(scan, mask)

    keys = list(feature_dict.keys())
    useful_keys = keys[keys.index('original_shape_Elongation'):]

    return feature_dict, useful_keys





import SimpleITK as sitk

def print_image_info(image, name):
    print(f"{name} information:")
    print(f"  Dimension: {image.GetDimension()}")
    print(f"  Size: {image.GetSize()}")
    print(f"  Spacing: {image.GetSpacing()}")
    print(f"  Origin: {image.GetOrigin()}")
    print(f"  Direction: {image.GetDirection()}")

print_image_info(image, 'ct')

print_image_info(mask, 'mask')





w = 7
start = 140
fig, axs = plt.subplots((350-start)//w, w)
for i in range(start, len(slices)):
    axs[(i-start)//w, (i-start)%w].imshow(slices[i], cmap='gray')
    axs[(i-start)//w, (i-start)%w].imshow(nifti_data[:, :, i], alpha=0.5, cmap='jet')  # Overlay mask
    axs[(i-start)//w, (i-start)%w].set_title('')  # Turn off title
    axs[(i-start)//w, (i-start)%w].set_xticks([])  # Turn off x-axis ticks
    axs[(i-start)//w, (i-start)%w].set_yticks([])  # Turn off y-axis ticks
    axs[(i-start)//w, (i-start)%w].set_xlabel('')  # Turn off x-axis label
    axs[(i-start)//w, (i-start)%w].set_ylabel('')  # Turn off y-axis label
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()




# Plot CT slices with mask overlay
w = 35
fig, axs = plt.subplots(10, w)
for i in range(len(slices)):
    axs[i//w, i%w].imshow(slices[i], cmap='gray')
    axs[i//w, i%w].imshow(nifti_data[:, :, i], alpha=0.5, cmap='jet')  # Overlay mask
    axs[i//w, i%w].set_title('')  # Turn off title
    axs[i//w, i%w].set_xticks([])  # Turn off x-axis ticks
    axs[i//w, i%w].set_yticks([])  # Turn off y-axis ticks
    axs[i//w, i%w].set_xlabel('')  # Turn off x-axis label
    axs[i//w, i%w].set_ylabel('')  # Turn off y-axis label
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()

for file_name in os.listdir(image_path):
    if file_name.endswith('.dcm'):
        # Load DICOM file
        slices = [pydicom.dcmread(os.path.join(image_path, file_name)) for file_name in sorted(os.listdir(image_path)) if file_name.endswith('.dcm')]
        ds = pydicom.dcmread(os.path.join(image_path, file_name))
        ct_image = ds.pixel_array.astype(np.float32)

        # Check consistency with mask
        assert ct_image.shape == mask.GetSize()[:2], "Dimensions of CT image and mask are not consistent."
        assert np.allclose(ds.PixelSpacing, mask.GetSpacing()), "Voxel sizes of CT image and mask are not consistent."
        assert np.allclose(ds.ImageOrientationPatient,
                           mask.GetDirection()), "Orientations of CT image and mask are not consistent."

        # Extract features from the DICOM file
        features = extractor.execute(ct_image, mask)

        # Access and process the extracted features
        print(f'Features extracted from {file_name}: {features}')








segmentation_mask_path_before = '/scratch/yx2432/dataset/CT/Mask/9001481772.nii.gz'

# check shape: ct_scan_before.GetSize()    segmentation_mask_before.GetSize()
mask = sitk.ReadImage(segmentation_mask_path_before)[:, :, 840:841]




# Load the image and mask (ROI) data
image_path = 'path_to_image.nii.gz'
mask_path = 'path_to_mask.nii.gz'

# Initialize the feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

# Extract features from the tumor region defined by the mask
features = extractor.execute(image_path, mask_path)

# Access and print the extracted features
for feature_name, value in features.items():
    print(f'{feature_name}: {value}')



scan = '/scratch/yx2432/dataset/CT/Image/9001481772/000840.dcm'
scan = sitk.ReadImage(scan)

# # Configure PyRadiomics settings
# params = {
#     'binWidth': 25,
#     'resampledPixelSpacing': [1, 1, 1],  # Resample voxel spacing to isotropic (optional)
#     'interpolator': 'sitkBSpline'  # Interpolator for resampling (optional)
# }

# resampled_scan, resampled_mask = imageoperations.resampleImage(scan, mask, resampledPixelSpacing=(1.0, 1.0, 1.0))
# pdb.set_trace()

# Initialize PyRadiomics feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

# Disable PyRadiomics logging (optional)
radiomics.logger.setLevel(radiomics.logging.ERROR)

# Extract features
feature_vector_before = extractor.execute(scan, mask)

# Display extracted features
for feature_name in feature_vector_before.keys():
    print(f"{feature_name}: {feature_vector_before[feature_name]}")


# segmentation_mask_array = sitk.GetArrayFromImage(segmentation_mask_before)
# use the already found target index 288
# segmentation_mask_before = segmentation_mask_before[:, :, 288]

# # np.unique(segmentation_mask_before[:, :, 288])
# # Iterate through slices to find the target slice idx
# for slice_index in tqdm(range(segmentation_mask_array.shape[0])):
#     slice_mask = segmentation_mask_array[slice_index, :, :]
#     unique_elements, element_counts = np.unique(slice_mask, return_counts=True)

#     if len(unique_elements) > 1 and element_counts[1] > 100:
#         pdb.set_trace()
#         break