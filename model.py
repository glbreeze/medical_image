import os
import pandas as pd

folder = '../dataset/CT/'
ImageFolder = 'image'
MaskFolder = 'mask'
TargetFolder = 'ct'

cl_features = pd.read_excel(os.path.join(folder, 'cl_features.xlsx'))
ml_features = pd.read_excel(os.path.join(folder, 'ml_features.xlsx'))


cl_features.rename(columns={'终点预测2分类': 'Group', '检查编号': 'imageName','试验系列名称':'set'}, inplace=True)
# cl_features['imageName'] = cl_features['imageName'].astype(str) + '.nii'

cl_features['set'] = cl_features['set'].str.lower()
cl_features['set'] = cl_features['set'].replace('lungmate-009', 'train_set')
cl_features['set'] = cl_features['set'].replace(['lungmate-001', 'lungmate-002'], 'validation_set')

column_order = ['Group', 'imageName'] + [col for col in cl_features.columns if col not in ['Group', 'imageName']]
cl_features = cl_features[column_order]
