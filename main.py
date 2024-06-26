import os, sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from utils import plot_cm, plot_roc, plot_feat_importance
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

folder = '../dataset/CT/'
ImageFolder = 'image'
MaskFolder = 'mask'
TargetFolder = 'ct'

cl_features = pd.read_excel(os.path.join(folder, 'cl_features_update.xlsx'))

# ====================== ml feature ======================
ml_features = pd.read_excel(os.path.join(folder, 'ml_features.xlsx'))
end_col = ml_features.columns.get_loc('original_shape_Elongation')
ml_features = ml_features.drop(ml_features.columns[1:end_col+1], axis=1)

ml_features = pd.merge(ml_features, cl_features[['imageName', 'Group']], on='imageName', how='inner')
column_order = ['Group', 'imageName'] + [col for col in ml_features.columns if col not in ['Group', 'imageName']]
ml_features = ml_features[column_order]

# ====================== ml feature selection ===================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split data into features (X) and target variable (y)
X = ml_features.drop(columns=['imageName', 'Group'])  # Exclude non-feature columns
y = ml_features['Group']

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, criterion='gini',)
clf.fit(X, y)

# Get feature importances
feature_importances = clf.feature_importances_

# Create a DataFrame to display feature importances
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Optionally, select top features based on importance threshold
top_features = importance_df[importance_df['Importance'] > 0.003]['Feature'].tolist()
selected_ml_features = ml_features[['imageName']+top_features]

selected_ml_features.to_csv('selected_ml_features.csv', index=False)

# ========================================== clinical feature ============================================
cl_features.rename(columns={'终点预测2分类': 'Group', '检查编号': 'imageName','试验系列名称':'set'}, inplace=True)
cl_features['set'] = cl_features['set'].str.lower()

column_order = ['Group', 'imageName'] + [col for col in cl_features.columns if col not in ['Group', 'imageName']]
cl_features = cl_features[column_order]

cl_features = cl_features[['Group', 'ORR', 'imageName', '年龄', '性别',	'set',	'是否吸烟',	'PDL1_expression', 'PDL1_number', '病理诊断_文本',
                           '病例完成治疗的周期数', '临床试验分期T', '临床试验N分期', '临床试验分期']]

cnt = cl_features.groupby(['set', 'Group']).size().reset_index(name='Count')
print(cnt)

# ================================== combine ml feature with cl features ==============================
all_features = pd.merge(cl_features, selected_ml_features, on='imageName', how='inner')
all_features['PDL1_number'] = all_features['PDL1_number'].replace('Unknown', np.nan)

categorical_features = ['性别', '是否吸烟', 'PDL1_expression', '病理诊断_文本', '临床试验分期']
for feature in ['性别', '是否吸烟', 'PDL1_expression', '病理诊断_文本', '临床试验分期']:
    # Convert categorical feature to integer codes
    all_features[feature] = all_features[feature].astype('category').cat.codes

# ================================== continuous feature for regression ==============================
trans = 'log1p'
# === which transformation
if trans == 'box-cox':
    from scipy.stats import boxcox
    data_positive = all_features['ORR']/100 + 1  # Adding 1 to ensure all values are positive
    transformed_data, lambda_value = boxcox(data_positive)
    all_features['y'] = ((all_features['ORR']/100 + 1)**lambda_value - 1)/lambda_value
elif trans == 'log1p':
    all_features['y'] = np.log1p(all_features['ORR']/100 + 1)

# ========================================= Classification =====================================
train_set = all_features[ all_features['set'].isin(['lungmate-009']) ]
val_set = all_features[ all_features['set'].isin(['lungmate-001', 'lungmate-002']) ]

X_train, y_train = train_set.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), train_set['Group']
X_val, y_val = val_set.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), val_set['Group']

cls = xgb.XGBClassifier()
cls.fit(X_train, y_train)

y_pred = cls.predict(X_val)
y_pred_p = cls.predict_proba(X_val)[:, 1]
accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)
auc = sklearn.metrics.roc_auc_score(y_val, y_pred_p)
# acc with new threshold
for th in np.linspace(0.1, 0.8, 20):
    accuracy = np.sum((y_pred_p>th) == y_val)/len(y_val)
    print('threshold {:.4f} acc {:.4f}'.format(th, accuracy))

print('accuracy: {:.4f}, auc: {:.4f}'.format(accuracy, auc))

# ===== feature importance and cm
plot_feat_importance(cls, X_train.columns, K=20)
plot_cm(y_val, y_pred)
plot_cm(y_val, y_pred_p>=0.75)
plot_roc(y_val, y_pred_p)

# ================================================== Regression ==================================================

train_set = all_features[ all_features['set'].isin(['lungmate-009']) ]
val_set = all_features[ all_features['set'].isin(['lungmate-001', 'lungmate-002']) ]

X_train, y_train = train_set.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), train_set['y']
X_val, y_val = val_set.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), val_set['y']


cls = xgb.XGBRegressor()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_val)

if trans == 'log1p':
    TH = np.log1p(1.3)
elif trans == 'box-cox':
    TH = (1.3**lambda_value - 1)/lambda_value


for th in np.linspace(0.1, 1, 20):
    y1_pred = (y_pred>th).astype(int)
    y1_val = (y_val > TH).astype(int)
    accuracy_score = np.sum(y1_pred == y1_val)/len(y1_val)
    print(th, 'acc', accuracy_score)

auc = sklearn.metrics.roc_auc_score((y_val>TH).astype(int), y_pred)

print('accuracy {:.4f}, auc: {:.4f}'.format(accuracy_score, auc))


# ============================================ New Split  + Classification=========================================
train_set = all_features[ all_features['set'].isin(['lungmate-009', 'lungmate-001']) ]
val_set = all_features[ all_features['set'].isin(['lungmate-002']) ]

X_train, y_train = train_set.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), train_set['Group']
X_val, y_val = val_set.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), val_set['Group']

cls = xgb.XGBClassifier()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_val)
y_pred_p = cls.predict_proba(X_val)[:,1]

accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)
auc = sklearn.metrics.roc_auc_score(y_val, y_pred_p)
print('Accuracy {:.4f}, AUC: {:.4f}'.format(accuracy, auc))

plot_cm(y_val, y_pred)
plot_feat_importance(cls, X_train.columns, K=20)
plot_roc(y_val, y_pred_p)

# ============================================ New Split  + Regression =========================================
train_set = all_features[ all_features['set'].isin(['lungmate-009', 'lungmate-001']) ]
val_set = all_features[ all_features['set'].isin(['lungmate-002']) ]

X_train, y_train = train_set.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), train_set['y']
X_val, y_val = val_set.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), val_set['y']

cls = xgb.XGBRegressor()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_val)

if trans == 'log1p':
    TH = np.log1p(1.3)
elif trans == 'box-cox':
    TH = (1.3**lambda_value - 1)/lambda_value

for th in np.linspace(0.1, 1, 20):
    y1_pred = (y_pred>th).astype(int)
    y1_val = (y_val > TH).astype(int)
    accuracy_score = np.sum(y1_pred == y1_val)/len(y1_val)
    print(th, 'acc', accuracy_score)

accuracy = np.sum((y_pred > TH).astype(int) == (y_val > TH).astype(int))/len(y_val)
auc = sklearn.metrics.roc_auc_score((y_val>TH).astype(int), y_pred)
print('accuracy {:.4f}, auc: {:.4f}'.format(accuracy, auc))

plot_cm((y_val > TH).astype(int), (y_pred > TH).astype(int))

plot_feat_importance(cls, X_train.columns, K=20)

plot_roc((y_val>TH).astype(int), y_pred)

# ============================================= random split + Classification ==========================================

X_train, X_val, y_train, y_val = train_test_split(all_features.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']),
                                                  all_features['Group'], test_size=0.3, random_state=21)

cls = xgb.XGBClassifier()
cls.fit(X_train, y_train)

y_pred = cls.predict(X_val)
y_pred_p = cls.predict_proba(X_val)[:, 1]
accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)
auc = sklearn.metrics.roc_auc_score(y_val, y_pred_p)

print('accuracy {} auc {}'.format(accuracy, auc))

plot_cm(y_val, y_pred)

plot_feat_importance(cls, X_train.columns, K=20)

plot_roc(y_val, y_pred_p)

# ============================================= random split + Regression ==========================================
X_train, X_val, y_train, y_val = train_test_split(all_features.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']),
                                                  all_features['Group'], test_size=0.3, random_state=21)

X_train, y_train = train_set.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), train_set['y']
X_val, y_val = val_set.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), val_set['y']

cls = xgb.XGBRegressor()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_val)

if trans == 'log1p':
    TH = np.log1p(1.3)
elif trans == 'box-cox':
    TH = (1.3**lambda_value - 1)/lambda_value

for th in np.linspace(0.1, 1, 20):
    y1_pred = (y_pred>th).astype(int)
    y1_val = (y_val > TH).astype(int)
    accuracy_score = np.sum(y1_pred == y1_val)/len(y1_val)
    print(th, 'acc', accuracy_score)

accuracy = np.sum((y_pred > TH).astype(int) == (y_val > TH).astype(int))/len(y_val)
auc = sklearn.metrics.roc_auc_score((y_val>TH).astype(int), y_pred)
print('accuracy {:.4f}, auc: {:.4f}'.format(accuracy, auc))

plot_cm((y_val > TH).astype(int), (y_pred > TH).astype(int))

plot_feat_importance(cls, X_train.columns, K=20)

plot_roc((y_val>TH).astype(int), y_pred)


# ================================================== oversample ==================================================
for feature in ['性别', '是否吸烟', 'PDL1_expression', '病理诊断_文本', '临床试验分期']:
    # Convert categorical feature to integer codes
    all_features[feature] = all_features[feature].astype('category').cat.codes
all_features['set'] = all_features['set'].replace('lungmate-009', 'train_set')
all_features['set'] = all_features['set'].replace(['lungmate-001', 'lungmate-002'], 'validation_set')
train_set = all_features[all_features['set'] == 'train_set']
val_set = all_features[all_features['set'] == 'validation_set']

val_counts = val_set[val_set['set'] == 'validation_set'].groupby(['Group', '病例完成治疗的周期数']).size()

train_counts = train_set.groupby(['Group', '病例完成治疗的周期数']).size()

# Calculate weight for each unique combination based on the desired proportion in the train dataset
weights = val_counts.div(train_counts, fill_value=0)
lower_cap, upper_cap = 0.5, 2  # Define the cap value
weights = weights.clip(lower = lower_cap, upper=upper_cap)

train_set['weight'] = train_set.apply(lambda row: weights.get((row['Group'], row['病例完成治疗的周期数']), 1), axis=1)

X_train, y_train = train_set.drop(columns=['imageName', 'Group', 'set', 'weight']), train_set['Group']
X_val, y_val = val_set.drop(columns=['imageName', 'Group', 'set']), val_set['Group']

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

cls = xgb.XGBClassifier()
cls.fit(X_train, y_train, sample_weight=train_set['weight'])

y_pred = cls.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

auc = roc_auc_score(y_val, y_pred)

print('accuracy: {:.4f}, auc: {:.4f}'.format(accuracy, auc))


##
neg, pos = np.bincount(y_train)

# Calculate scale_pos_weight value
scale_pos_weight = neg / pos

# Initialize XGBoost classifier with scale_pos_weight parameter
model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)





