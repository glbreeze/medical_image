# -*- coding: utf-8 -*-
import os, sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from utils import plot_cm, plot_roc, plot_feat_importance
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

folder = '../dataset/CT/'
n_feat = 50

# ============ read data ============
train_df = pd.read_csv(os.path.join(folder, 'all_features.csv'))
test_df = pd.read_csv(os.path.join(folder, 'all_features_241.csv'))
test_df['PDL1_number'] = test_df['PDL1_number'].replace('<1', '0')
test_df['PDL1_number'] = test_df['PDL1_number'].replace('1-49', '40')

# ========= data process ===========
for df in [train_df, test_df]:
    df['PDL1_number'] = df['PDL1_number'].replace('Unknown', np.nan)
    df['PDL1_number'] = pd.to_numeric(df['PDL1_number'])
    df['PDL1_number_none'] = df['PDL1_number'].isna()

    # continuous feature for regression
    trans = 'log1p'
    if trans == 'box-cox':
        from scipy.stats import boxcox

        data_positive = df['ORR'] / 100 + 1  # Adding 1 to ensure all values are positive
        transformed_data, lambda_value = boxcox(data_positive)
        df['y'] = ((df['ORR'] / 100 + 1) ** lambda_value - 1) / lambda_value
    elif trans == 'log1p':
        df['y'] = np.log1p(df['ORR'] / 100 + 1)


# ========== data processing for the pipeline
categorical_cols = ['性别', '是否吸烟', 'PDL1_expression', '病理诊断_文本', '临床试验分期', '临床试验分期T', 'PDL1_number_none']
numerical_cols = [col for col in df.columns if col not in categorical_cols + ['imageName', 'Group', 'set', 'ORR', 'y']]

def fill_missing_with_category(X):
    return X.fillna('Missing')

numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('fillna', FunctionTransformer(fill_missing_with_category, validate=False)),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

for df in [train_df, test_df]:
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        print('col {} is str'.format(col))
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col])
        print('col {} is numeric'.format(col))

X_train, y_train = train_df.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), train_df['Group']
X_val, y_val = test_df.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), test_df['Group']

model = RandomForestClassifier(n_estimators=100, random_state=42)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
y_pred_p = clf.predict_proba(X_val)[:, 1]

# Evaluate the model
print("Accuracy:", accuracy_score(y_val, y_pred))
print("AUC", sklearn.metrics.roc_auc_score(y_val, y_pred_p))
print("Classification Report:")
print(sklearn.metrics.classification_report(y_val, y_pred))


# ====== regression ========
X_train, y_train = train_df.drop(columns=['imageName', 'Group', 'set', 'ORR', 'Group']), train_df['y']
X_val, y_val = test_df.drop(columns=['imageName', 'Group', 'set', 'ORR', 'Group']), test_df['y']
model = RandomForestRegressor(n_estimators=100, random_state=42)

clf = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)

if trans == 'log1p':
    TH = np.log1p(1.3)
elif trans == 'box-cox':
    TH = (1.3 ** lambda_value - 1) / lambda_value

auc = sklearn.metrics.roc_auc_score((y_val > TH).astype(int), y_pred)
print('auc is {}'.format(auc))

