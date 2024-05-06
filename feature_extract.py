import radiomics
from radiomics import featureextractor
import pandas as pd
import os
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor

folder = '../dataset/CT/'
ImageFolder = 'image'
MaskFolder = 'mask'
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

image_types = {'Original': {},
               'LoG': {'sigma': [1.0, 2.0]}, # Features extracted from Laplacian of Gaussian filtered images
               'Wavelet': {'name': 'db', 'level': 2},
               }

extractor = featureextractor.RadiomicsFeatureExtractor(**params)
extractor.enableImageTypes(**image_types)


df = pd.DataFrame()
for imageName in os.listdir(os.path.join(folder, TargetFolder)):
        ImagePath = os.path.join(folder, TargetFolder, imageName)
        MaskPath = os.path.join(folder, MaskFolder, imageName)
        if os.path.exists(MaskPath):
            try:
                featureVector = extractor.execute(ImagePath, MaskPath)
                df_add = pd.DataFrame.from_dict(featureVector.values()).T
                df_add.columns = featureVector.keys()
                df_add.insert(0, 'imageName', imageName.split('.')[0])
                df = pd.concat([df, df_add])
            except Exception as e:
                print(f"extracting feature from {ImagePath} and {MaskPath}, Error {e}")
                continue
        else:
            print('The File for mask {} does NOT exist'.format(MaskPath))


result_file = os.path.join(folder, 'ml_features.xlsx')
df.to_excel(result_file, index=False)
print("Have stored file to {}".format(result_file))