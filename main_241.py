# -*- coding: utf-8 -*-
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
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict

folder = '../dataset/CT/'

def find_key_by_value(d, value):
    for key, val in d.items():
        if abs(val - value) <= 1e-8:
            return key
    return None

# ========================================== clinical feature ============================================
cl_features = pd.read_excel(os.path.join(folder, '影像组学总库2024-6-6.xlsx'))
cl_columns = ['ORR', 'RECIST', '检查编号', '筛选号', '试验系列名称', '性别', '就诊年龄', '是否吸烟', 'Drug', '病理诊断_文本',
              '病例完成治疗的周期数', '临床试验分期T', '临床试验N分期', '临床试验分期',
              'PDL1_expression', 'PDL1_number', '终点预测2分类', '新辅助治疗开始时间', '新辅助治疗方案',
              '手术', '残留肿瘤细胞', 'MPR', 'pCR', '首次治疗时间', '随访日期', '是否疾病进展或复发转移', '复发转移时间', '复发与转移部位',
              '是否死亡', '死亡时间', '死亡原因', '临床试验分期M', '最长径（淋巴结为最短经）', '每日吸烟数量（支）', '吸烟时长（年）', '驱动基因突变',
              '增强', '检查部位', '检查所见', '检查结论', '数据库来源']
cl_features = cl_features[cl_columns]

cl_features.rename(
    columns={'终点预测2分类': 'Group', '检查编号': 'imageName', '试验系列名称': 'set', '就诊年龄': '年龄', '每日吸烟数量（支）': '每日吸烟数量'},
    inplace=True)
cl_features['set'] = cl_features['set'].str.lower()

column_order = ['Group', 'imageName'] + [col for col in cl_features.columns if col not in ['Group', 'imageName']]
cl_features = cl_features[column_order]

cl_features = cl_features[['Group', 'ORR', 'imageName', '年龄', '性别', 'set', '是否吸烟', '每日吸烟数量',
                           'PDL1_expression', 'PDL1_number', '病理诊断_文本', '病例完成治疗的周期数', '临床试验分期T', '临床试验N分期', '临床试验分期']]
cl_features.loc[cl_features['是否吸烟'] == 0, '每日吸烟数量'] = 0
cl_features = cl_features[cl_features['Group'].notna()]

cnt = cl_features.groupby(['set', 'Group']).size().reset_index(name='Count')
print(cnt)

# ====================== ml feature ======================
ml_features = pd.read_excel(os.path.join(folder, 'ml_features_241.xlsx'))
end_col = ml_features.columns.get_loc('original_shape_Elongation')
ml_features = ml_features.drop(ml_features.columns[1:end_col + 1], axis=1)

train_df = pd.read_csv(os.path.join(folder, 'all_features.csv'))
selected_columns = [col for col in train_df.columns if col in ml_features.columns]
selected_ml_features = ml_features[selected_columns]

# ================================== combine ml feature with cl features ==============================
all_features = pd.merge(cl_features, selected_ml_features, on='imageName', how='inner')
all_features.to_csv(os.path.join(folder, 'all_features_241.csv'), index=False)

train_df = pd.read_csv(os.path.join(folder, 'all_features.csv'))
test_df = pd.read_csv(os.path.join(folder, 'all_features_241.csv'))
test_df['PDL1_number'] = test_df['PDL1_number'].replace('<1', '0')
test_df['PDL1_number'] = test_df['PDL1_number'].replace('1-49', '40')

num_ml_feat = 50
ml_feat_start = train_df.columns.get_loc('wavelet2-LLL_gldm_LargeDependenceHighGrayLevelEmphasis')
train_df = train_df.drop(train_df.columns[ml_feat_start+num_ml_feat:], axis=1)
test_df = test_df.drop(test_df.columns[ml_feat_start+num_ml_feat:], axis=1)
# ================================== Process features  ==============================

for df in [train_df, test_df]:
    df['PDL1_number'] = df['PDL1_number'].replace('Unknown', np.nan)
    df['PDL1_number'] = pd.to_numeric(df['PDL1_number'])

    # Convert categorical feature to integer codes
    categorical_features = ['性别', '是否吸烟', 'PDL1_expression', '病理诊断_文本', '临床试验分期', '临床试验分期T']
    for feature in categorical_features:
        df[feature] = df[feature].astype('category').cat.codes

    # continuous feature for regression
    trans = 'log1p'
    if trans == 'box-cox':
        from scipy.stats import boxcox

        data_positive = df['ORR'] / 100 + 1  # Adding 1 to ensure all values are positive
        transformed_data, lambda_value = boxcox(data_positive)
        df['y'] = ((df['ORR'] / 100 + 1) ** lambda_value - 1) / lambda_value
    elif trans == 'log1p':
        df['y'] = np.log1p(df['ORR'] / 100 + 1)

# ========================================= Classification =====================================

X_train, y_train = train_df.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), train_df['Group']
X_val, y_val = test_df.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), test_df['Group']

cls = xgb.XGBClassifier()

# Grid search of the hyper-parameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1]
}
grid_search = GridSearchCV(estimator=cls, param_grid=param_grid, cv=5, scoring='roc_auc', verbose=1,
                           n_jobs=-1)  # roc_auc_ovr
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# best_model = grid_search.best_estimator_
best_model = xgb.XGBClassifier(**grid_search.best_params_)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_val)
y_pred_p = best_model.predict_proba(X_val)[:, 1]
accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)
auc = sklearn.metrics.roc_auc_score(y_val, y_pred_p)
print('accuracy: {:.4f}, auc: {:.4f}'.format(accuracy, auc))


# ==== now plot the training roc
best_model = xgb.XGBClassifier(**grid_search.best_params_)
y_pred_p = cross_val_predict(best_model, X_train, y_train, cv=5, method='predict_proba')
y_pred = np.argmax(y_pred_p, axis=1)
y_pred_p = y_pred_p[:, 1]
accuracy = sklearn.metrics.accuracy_score(y_train, y_pred)
auc = sklearn.metrics.roc_auc_score(y_train, y_pred_p)
print('accuracy: {:.4f}, auc: {:.4f}'.format(accuracy, auc))



# ===== feature importance and cm
plot_feat_importance(best_model, X_train.columns, K=20)
plot_cm(y_val, y_pred)
plot_cm(y_val, y_pred_p >= 0.75)
plot_roc(y_val, y_pred_p)


# ========================================= Classification with subset  =====================================
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict

del_col = ['wavelet2-LLL_gldm_LargeDependenceLowGrayLevelEmphasis']
out_dict = {}
for set_name in ['lungmate-001', 'lungmate-002', 'lungmate-009']:
    train_df_sub = train_df[train_df['set'] != set_name]
    X_train, y_train = train_df_sub.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), train_df_sub['Group']
    X_val, y_val = test_df.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), test_df['Group']

    cls = xgb.XGBClassifier()
    cls.fit(X_train, y_train)

    y_pred = cls.predict(X_val)
    y_pred_p = cls.predict_proba(X_val)[:, 1]
    accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)
    auc = sklearn.metrics.roc_auc_score(y_val, y_pred_p)
    print('get rid of {}, accuracy: {:.4f}, auc: {:.4f}'.format(set_name, accuracy, auc))
    out_dict['col'] = [accuracy, auc]

# ================================================== Regression choose subset ==================================================
out_dict = {}

for set_name in ['lungmate-001', 'lungmate-002', 'lungmate-009', 'none']:
    train_df_sub = train_df[train_df['set'] != set_name]
    X_train, y_train = train_df_sub.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), train_df_sub['Group']
    X_val, y_val = test_df.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), test_df['Group']

    cls = xgb.XGBRegressor()
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_val)

    if trans == 'log1p':
        TH = np.log1p(1.3)
    elif trans == 'box-cox':
        TH = (1.3 ** lambda_value - 1) / lambda_value

    auc = sklearn.metrics.roc_auc_score((y_val > TH).astype(int), y_pred)
    print('get rid of {}, auc is {}'.format(set_name, auc))
    out_dict[set_name] = auc

max(out_dict.values())
plt.scatter(test_df['ORR'], y_pred)

# ========================================= Classification =====================================
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict

del_col = ['wavelet2-LLL_gldm_LargeDependenceLowGrayLevelEmphasis']
out_dict = {}
for col in [col for col in train_df.columns if col not in ['imageName', 'Group', 'set', 'ORR', 'y', 'Group']]:
    X_train, y_train = train_df.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y', col] + del_col), train_df['Group']
    X_val, y_val = test_df.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y', col] + del_col), test_df['Group']

    cls = xgb.XGBClassifier()
    cls.fit(X_train, y_train)

    y_pred = cls.predict(X_val)
    y_pred_p = cls.predict_proba(X_val)[:, 1]
    accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)
    auc = sklearn.metrics.roc_auc_score(y_val, y_pred_p)
    print('get rid of {}, accuracy: {:.4f}, auc: {:.4f}'.format(col, accuracy, auc))
    out_dict['col'] = [accuracy, auc]

# ================================================== Regression ==================================================
train_df_sub = train_df[train_df['set'] != 'lungmate-002']

del_cols = []
for i in range(20):
    out_dict = {}
    for col in [col for col in train_df.columns if col not in ['imageName', 'Group', 'set', 'ORR', 'y', 'Group'] + del_cols]:
        X_train, y_train = train_df_sub.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y', col] + del_cols), train_df_sub['y']
        X_val, y_val = test_df.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y', col]+del_cols), test_df['y']

        cls = xgb.XGBRegressor()
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_val)

        if trans == 'log1p':
            TH = np.log1p(1.3)
        elif trans == 'box-cox':
            TH = (1.3 ** lambda_value - 1) / lambda_value

        auc = sklearn.metrics.roc_auc_score((y_val > TH).astype(int), y_pred)
        print('get rid of {}, auc is {}'.format(col, auc))
        out_dict[col] = auc

    del_col = find_key_by_value(out_dict, max(out_dict.values()))
    print('\n====>Step: {}, del col: {}, the max auc is: {:.4f}'.format(i, del_col, max(out_dict.values())))
    del_cols.append(del_col)


# ========================================= Final regression =====================================
X_train, y_train = train_df_sub.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y'] + del_cols), train_df_sub['y']
X_val, y_val = test_df.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']+del_cols), test_df['y']
cls = xgb.XGBRegressor()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_val)

if trans == 'log1p':
    TH = np.log1p(1.3)
elif trans == 'box-cox':
    TH = (1.3 ** lambda_value - 1) / lambda_value

accuracy = np.sum((y_pred > TH-0.1).astype(int) == (y_val > TH).astype(int))/len(y_val)
auc = sklearn.metrics.roc_auc_score((y_val>TH).astype(int), y_pred)
print('accuracy {:.4f}, auc: {:.4f}'.format(accuracy, auc))

plot_cm((y_val > TH-0.1).astype(int), (y_pred > TH-0.1).astype(int))

plot_feat_importance(cls, X_train.columns, K=20)

plot_roc((y_val>TH).astype(int), y_pred)


from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict

X_train, y_train = train_df_sub.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y', col] + del_cols), train_df_sub['y']
X_val, y_val = test_df.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y', col]+del_cols), test_df['y']

cls = xgb.XGBRegressor()

# Grid search of the hyper-parameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1]
}
grid_search = GridSearchCV(estimator=cls, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1) # roc_auc_ovr
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# best_model = grid_search.best_estimator_
best_model = xgb.XGBClassifier(**grid_search.best_params_)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_val)
y_pred_p = best_model.predict_proba(X_val)[:, 1]
accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)
auc = sklearn.metrics.roc_auc_score(y_val, y_pred_p)
print('accuracy: {:.4f}, auc: {:.4f}'.format(accuracy, auc))

# ==== now plot the training roc
best_model = xgb.XGBClassifier(**grid_search.best_params_)
y_pred_p = cross_val_predict(best_model, X_train, y_train, cv=5, method='predict_proba')
y_pred = np.argmax(y_pred_p, axis=1)
y_pred_p = y_pred_p[:,1]
accuracy = sklearn.metrics.accuracy_score(y_train, y_pred)
auc = sklearn.metrics.roc_auc_score(y_train, y_pred_p)
print('accuracy: {:.4f}, auc: {:.4f}'.format(accuracy, auc))

# ===== feature importance and cm
plot_feat_importance(cls, X_train.columns, K=20)
plot_cm(y_val, y_pred)
plot_cm(y_val, y_pred_p>=0.75)
plot_roc(y_val, y_pred_p)



if trans == 'log1p':
    TH = np.log1p(1.3)
elif trans == 'box-cox':
    TH = (1.3**lambda_value - 1)/lambda_value

accuracy = np.sum((y_pred > TH).astype(int) == (y_val > TH).astype(int))/len(y_val)
auc = sklearn.metrics.roc_auc_score((y_val>TH).astype(int), y_pred)
print('accuracy {:.4f}, auc: {:.4f}'.format(accuracy, auc))

plot_cm((y_val > TH).astype(int), (y_pred > TH).astype(int))

plot_feat_importance(cls, X_train.columns, K=20)

plot_roc((y_val>TH).astype(int), y_pred)





for th in np.linspace(0.1, 1, 20):
    y1_pred = (y_pred > th).astype(int)
    y1_val = (y_val > TH).astype(int)
    accuracy_score = np.sum(y1_pred == y1_val) / len(y1_val)
    print(th, 'acc', accuracy_score)

print('accuracy {:.4f}, auc: {:.4f}'.format(accuracy_score, auc))

