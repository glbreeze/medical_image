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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


folder = '../dataset/CT/'
n_feat = 50

# ============ read data ============
train_df = pd.read_csv(os.path.join(folder, 'all_features.csv'))
test_df = pd.read_csv(os.path.join(folder, 'all_features_241.csv'))
test_df['PDL1_number'] = test_df['PDL1_number'].replace('<1', '0')
test_df['PDL1_number'] = test_df['PDL1_number'].replace('1-49', '40')

# ========= data process ===========
categorical_cols = ['性别', '是否吸烟', 'PDL1_expression', '病理诊断_文本', '临床试验分期', '临床试验分期T', 'PDL1_number_none']
numerical_cols = [col for col in train_df.columns if col not in categorical_cols + ['imageName', 'Group', 'set', 'ORR', 'y']]
X_train = train_df[categorical_cols + numerical_cols].copy()
X_test = test_df[categorical_cols + numerical_cols].copy()
for i, df in enumerate([X_train, X_test]):
    df['PDL1_number'] = df['PDL1_number'].replace('Unknown', np.nan)
    df['PDL1_number'] = pd.to_numeric(df['PDL1_number'])
    df['PDL1_number_none'] = df['PDL1_number'].apply(lambda x: 0 if pd.isna(x) else 1)

    for col in categorical_cols:
        df[col] = df[col].astype(str)
        print('col {} is str'.format(col))
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col])
        print('col {} is numeric'.format(col))

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    if i == 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        X_train = preprocessor.fit_transform(df)
    else:
        X_test = preprocessor.transform(df)

# ========== define data loader
Xtrain_tensor = torch.tensor(X_train, dtype=torch.float32)
ytrain_tensor = torch.tensor(train_df['ORR'].values, dtype=torch.float32).unsqueeze(1)
Xtest_tensor = torch.tensor(X_test, dtype=torch.float32)
ytest_tensor = torch.tensor(test_df['ORR'].values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(Xtrain_tensor, ytrain_tensor)
test_dataset = TensorDataset(Xtest_tensor, ytest_tensor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ======== define model
class MLP(nn.Module):
    def __init__(self, input_dim, arch='256_256_256', bn='t'):
        super(MLP, self).__init__()

        module_list = []
        for i, hidden_size in enumerate(arch.split('_')):
            hidden_size = int(hidden_size)
            module_list.append(nn.Linear(input_dim, hidden_size))
            if bn == 't':
                module_list.append(nn.BatchNorm1d(hidden_size, affine=False))
            elif bn == 'p':
                module_list.append(nn.BatchNorm1d(hidden_size, affine=True))
            module_list.append(nn.ReLU())
            input_dim = hidden_size

        self.backbone = nn.Sequential(*module_list)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        return out


input_dim = X_train.shape[1]
model = MLP(input_dim)

# ========= train the model

def validate_model(test_loader, epoch=0):
    model.eval()
    test_loss = 0.0
    y_pred, y_label = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            y_pred.append(outputs)
            y_label.append(targets)
    y_pred = torch.concatenate(y_pred, dim=0).flatten().cpu().numpy()
    y_label = torch.concatenate(y_label, dim=0).flatten().cpu().numpy()

    auc = sklearn.metrics.roc_auc_score((y_label > 30).astype(int), y_pred)
    print('Epoch:{}, Test AUC:{:.4f}'.format(epoch, auc))
    return auc


criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

epochs = 100
best_auc = 0.0
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')
    test_auc = validate_model(test_loader, epoch=epoch)

    if test_auc > best_auc:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, 'best_ckpt.pth')








