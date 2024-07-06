# -*- coding: utf-8 -*-
import os, sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


folder = '../dataset/CT/'
n_feat = 50

# ============ read data ============
all_df = pd.read_csv(os.path.join(folder, 'all_features.csv'))

# ========= data process ===========
def process_data(df):
    categorical_cols = ['性别', '是否吸烟', 'PDL1_expression', '病理诊断_文本', '临床试验分期', '临床试验分期T', 'PDL1_number_none']
    numerical_cols = [col for col in df.columns if col not in categorical_cols + ['imageName', 'Group', 'set', 'ORR', 'y']]
    df['PDL1_number'] = df['PDL1_number'].replace('missing', np.nan)
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

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    return df, preprocessor

# ===== split and transform data

def split_process_data(all_data, preprocessor, split_type='19_2'):
    if split_type == '9_12':
        train_set = all_data[all_data['set'].isin(['lungmate-009'])]
        val_set = all_data[all_data['set'].isin(['lungmate-001', 'lungmate-002'])]

        X_train, y_train = train_set.drop(columns=['imageName', 'Group', 'set', 'ORR']), train_set[['Group']]
        X_val, y_val = val_set.drop(columns=['imageName', 'Group', 'set', 'ORR']), val_set[['Group']]

    elif split_type == '19_2':
        train_set = all_data[all_data['set'].isin(['lungmate-009', 'lungmate-001'])]
        val_set = all_data[all_data['set'].isin(['lungmate-002'])]

        X_train, y_train = train_set.drop(columns=['imageName', 'Group', 'set', 'ORR']), train_set[['Group']]
        X_val, y_val = val_set.drop(columns=['imageName', 'Group', 'set', 'ORR']), val_set[['Group']]

    elif split_type == 'random':
        X_train, X_val, y_train, y_val = train_test_split(all_data.drop(columns=['imageName', 'Group', 'set', 'ORR']),
                                                          all_data[['Group']], test_size=0.3, random_state=21)

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)

    return X_train, y_train['Group'].values, X_val, y_val['Group'].values


# ==== get data loader

def get_loader(X_train, y_train, X_val, y_val):
    Xtrain_tensor = torch.tensor(X_train, dtype=torch.float32)
    ytrain_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    Xtest_tensor = torch.tensor(X_val, dtype=torch.float32)
    ytest_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(Xtrain_tensor, ytrain_tensor)
    test_dataset = TensorDataset(Xtest_tensor, ytest_tensor)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return  train_loader, test_loader

# ======== define model
class MLP(nn.Module):
    def __init__(self, input_dim, arch='256_256', bn='p'):
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
        return torch.sigmoid(out)


# ========= train the model

def validate_model(model, test_loader, epoch=0):
    model.eval()
    y_pred, y_label = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            y_pred.append(outputs)
            y_label.append(targets)
    y_pred = torch.concatenate(y_pred, dim=0).flatten().cpu().numpy()
    y_label = torch.concatenate(y_label, dim=0).flatten().cpu().numpy()

    acc = np.sum( (y_pred>=0.5).astype(np.float32) == y_label )/len(y_label)
    auc = sklearn.metrics.roc_auc_score(y_label, y_pred)
    print('Epoch:{}, Test Acc: {:.4f}, Test AUC:{:.4f}'.format(epoch, acc, auc))
    return auc, y_pred, y_label


def trainer(model, optimizer, train_loader, test_loader, epoch=50):
    criterion = nn.BCELoss()
    epochs = 40
    best_auc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_cor = 0
        num_all = 0
        for inputs, targets in train_loader:
            if len(targets) <= 2:
                continue
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_cor += torch.sum( (outputs >= 0.5).float() == targets ).item()
            num_all += len(targets)

        if epoch % 2 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader)}')
            print(f'Epoch {epoch + 1}/{epochs}, Train Acc: {num_cor / num_all :.4f}')
            test_auc, y_pred, y_label = validate_model(model, test_loader, epoch=epoch)

        if False and (test_auc > best_auc):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, 'best_ckpt.pth')

all_data, preprocessor = process_data(all_df)
X_train, y_train, X_val, y_val = split_process_data(all_data, preprocessor=preprocessor, split_type='9_12')
train_loader, test_loader = get_loader(X_train, y_train, X_val, y_val)


input_dim = train_loader.dataset.__getitem__(0)[0].shape[0]
model = MLP(input_dim)
optimizer = optim.SGD(model.parameters(), lr=0.001)
trainer(model, optimizer, train_loader, test_loader)


# =============================== Cross validate Train ===============================
cv = 5
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=17)

# == process all training data
train_set = all_data[ all_data['set'].isin(['lungmate-009', 'lungmate-001']) ]
X_train, y_train = train_set.drop(columns=['imageName', 'Group', 'set', 'ORR']), train_set[['Group']]

X = preprocessor.fit_transform(X_train)
y = y_train['Group'].values

# == split the data
y_preds = []
y_labels = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_loader, test_loader = get_loader(X_train, y_train, X_test, y_test)

    # == init model
    input_dim = train_loader.dataset.__getitem__(0)[0].shape[0]
    model = MLP(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    trainer(model, optimizer, train_loader, test_loader, epoch=50)
    _, y_pred, y_label = validate_model(model, test_loader)  # tensor
    y_preds.append(y_pred)
    y_labels.append(y_label)

y_preds = np.concatenate(y_preds)
y_labels = np.concatenate(y_labels)
cv_acc = np.sum( (y_preds>=0.5).astype(np.float32) == y_labels )/len(y_labels)
cv_auc = sklearn.metrics.roc_auc_score(y_labels, y_preds)
print('=====>Cross validation Acc: {:.4f},  AUC:{:.4f}'.format(cv_acc, cv_auc))




