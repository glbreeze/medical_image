# -*- coding: utf-8 -*-
from utils import plot_cm, plot_roc, plot_feat_importance, catboost_cross_val_predict

import os, sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from boruta import BorutaPy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

folder = '../dataset/CT/'
n_feat = 30
categorical_features = ['性别', '是否吸烟', 'PDL1_expression', '病理诊断_文本', '临床试验分期', '临床试验分期T']

all_result ={}

# ========================================== clinical feature ============================================
cl_features = pd.read_excel(os.path.join(folder, '影像组学总库2024-6-6.xlsx'))
cl_columns = ['ORR', 'RECIST', '检查编号', '筛选号', '试验系列名称', '性别', '就诊年龄', '是否吸烟', 'Drug', '病理诊断_文本',
              '病例完成治疗的周期数', '临床试验分期T', '临床试验N分期', '临床试验分期', 'PDL1_expression', 'PDL1_number',
              '终点预测2分类', '新辅助治疗开始时间', '新辅助治疗方案', '手术', '残留肿瘤细胞', 'MPR', 'pCR',  '首次治疗时间',
              '随访日期', '是否疾病进展或复发转移', '复发转移时间', '复发与转移部位', '是否死亡', '死亡时间', '死亡原因',
              '临床试验分期M', '最长径（淋巴结为最短经）', '是否吸烟2', '每日吸烟数量（支）', '吸烟时长（年）', '驱动基因突变',
              '增强',  '检查部位', '检查所见', '检查结论', '数据库来源']
cl_features = cl_features[cl_columns]

cl_features.rename(columns={'终点预测2分类': 'Group', '检查编号': 'imageName','试验系列名称':'set', '就诊年龄':'年龄', '每日吸烟数量（支）': '每日吸烟数量'}, inplace=True)
cl_features['set'] = cl_features['set'].str.lower()

cl_features = cl_features[['Group', 'ORR', 'imageName', '年龄', '性别',	'set',	'是否吸烟', '每日吸烟数量' , 'PDL1_expression',
                           'PDL1_number', '病理诊断_文本', '病例完成治疗的周期数', '临床试验分期T', '临床试验N分期', '临床试验分期']]

cl_features.loc[cl_features['是否吸烟'] == 0, '每日吸烟数量'] = 0

replacement_dict1 = {'1a': 1, '1b': 1, '1c': 1, '2a': 2, '2b': 2, '4':4}
cl_features['临床试验分期T'] = cl_features['临床试验分期T'].replace(replacement_dict1)

replacement_dict2 = {'<1': 0, '1-49': 25}
cl_features['PDL1_number'] = cl_features['PDL1_number'].replace(replacement_dict2)
cl_features.replace(['Unknown', 'Others'], 'missing', inplace=True)
cl_features = cl_features[cl_features['Group'].notna()]

# ====================== ml feature and feature selection======================
ml_features = pd.read_excel(os.path.join(folder, 'ml_feature_2.xlsx'))
ml_features = pd.merge(cl_features[cl_features['set'].isin(['lungmate-009', 'lungmate-002', 'lungmate-001'])]['imageName'],
                       ml_features, on='imageName', how='inner')

X = ml_features.drop(columns=['imageName', 'Group'])  # Exclude non-feature columns
y = ml_features['Group']

method = 'rf'
if method == 'rf':
    clf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=21)
    clf.fit(X, y)

    feature_importances = clf.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    threshold = np.partition(feature_importances, -n_feat)[-n_feat]
    top_features = importance_df[importance_df['Importance'] > threshold]['Feature'].tolist()
    selected_ml_features = ml_features[['imageName']+top_features]
else:
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)
    boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=42)  # n_estimators='auto'

    np.int = np.int32
    np.float = np.float64
    np.bool = np.bool_
    boruta_selector.fit(np.array(X), np.array(y))
    selected_features = pd.DataFrame({'Feature': list(X.columns),
                                      'Ranking': boruta_selector.ranking_}).sort_values(by='Ranking')
    top_features = list(selected_features[selected_features["Ranking"] < n_feat]['Feature'])
    selected_ml_features = ml_features[['imageName'] + top_features]


# ================================== combine ml feature with cl features ==============================
all_features = pd.merge(cl_features[cl_features['set'].isin(['lungmate-009', 'lungmate-002', 'lungmate-001'])],
                        selected_ml_features, on='imageName', how='inner')
all_features['是否吸烟'] = all_features['是否吸烟'].astype('Int64')
all_features['是否吸烟'] = all_features['是否吸烟'].fillna(-1)
all_features['临床试验分期T'] = all_features['临床试验分期T'].astype('Int64')

if False:
    all_features.to_csv(os.path.join(folder, 'all_features.csv'), index=False)

for col in all_features.columns:
    if all_features[col].isna().any():
        print(col, 'has na',  all_features[col].isna().any())

# ================================== Preprocess features for XGBoost ==============================
def process_df(all_features):
    all_features['PDL1_number'] = all_features['PDL1_number'].replace('missing', np.nan)
    all_features['PDL1_number'] = pd.to_numeric(all_features['PDL1_number'])

    for feature in categorical_features:
        all_features[feature] = all_features[feature].astype('category').cat.codes

    trans = 'log1p'
    # === which transformation
    if trans == 'box-cox':
        from scipy.stats import boxcox
        data_positive = all_features['ORR']/100 + 1  # Adding 1 to ensure all values are positive
        transformed_data, lambda_value = boxcox(data_positive)
        all_features['y'] = ((all_features['ORR']/100 + 1)**lambda_value - 1)/lambda_value
    elif trans == 'log1p':
        all_features['y'] = np.log1p(all_features['ORR']/100 + 1)

    return all_features


def load_data(all_data, split_type):
    if split_type == '9_12':
        train_set = all_data[ all_data['set'].isin(['lungmate-009']) ]
        val_set = all_data[ all_data['set'].isin(['lungmate-001', 'lungmate-002']) ]

        X_train, y_train = train_set.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), train_set['Group']
        X_val, y_val = val_set.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), val_set['Group']

    elif split_type == '19_2':
        train_set = all_data[ all_data['set'].isin(['lungmate-009', 'lungmate-001']) ]
        val_set = all_data[ all_data['set'].isin(['lungmate-002']) ]

        X_train, y_train = train_set.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), train_set['Group']
        X_val, y_val = val_set.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']), val_set['Group']

    elif split_type == 'random':
        X_train, X_val, y_train, y_val = train_test_split(all_data.drop(columns=['imageName', 'Group', 'set', 'ORR', 'y']),
                                                          all_data['Group'], test_size=0.3, random_state=21)
    return X_train, y_train, X_val, y_val


# ========================================= XGBoost Classification =====================================

split_type = '19_2'
all_data = process_df(all_features)
X_train, y_train, X_val, y_val = load_data(all_data, split_type)

cls = xgb.XGBClassifier()

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1]
}
grid_search = GridSearchCV(estimator=cls, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1) # roc_auc
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
print('Test Accuracy: {:.4f}, AUC: {:.4f}'.format(accuracy, auc))

all_result['xgboost'] = [y_val, y_pred_p]

plot_feat_importance(best_model, X_train.columns, K=20)
plot_cm(y_val, y_pred_p>=0.5)
plot_roc(y_val, y_pred_p)

# =========== training roc ===========
best_model = xgb.XGBClassifier(**grid_search.best_params_)
y_pred_p = cross_val_predict(best_model, X_train, y_train, cv=37, method='predict_proba')
y_pred = np.argmax(y_pred_p, axis=1)
y_pred_p = y_pred_p[:,1]
accuracy = sklearn.metrics.accuracy_score(y_train, y_pred)
auc = sklearn.metrics.roc_auc_score(y_train, y_pred_p)
print('Train accuracy: {:.4f}, Train AUC: {:.4f}'.format(accuracy, auc))

plot_cm(y_train, y_pred_p>=0.5)
plot_roc(y_train, y_pred_p)


# ========================================= LightGBM Classification =====================================
all_data = process_df(all_features)
X_train, y_train, X_val, y_val = load_data(all_data, split_type)

lgbm = lgb.LGBMClassifier(random_state=42)  # num_iterations=50

param_grid = {
    "learning_rate": [0.001],
    "num_leaves": [31, 63, 127],
    "max_depth": [-1, 3, 5],
    "subsample": [0.8, 1.0]
}
grid_search = GridSearchCV(lgbm, param_grid, cv=5, scoring="roc_auc")
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

best_lgbm = lgb.LGBMClassifier(random_state=42, **grid_search.best_params_)
best_lgbm.fit(X_train, y_train, categorical_feature=categorical_features)

y_pred = best_lgbm.predict(X_val)
y_pred_p = best_lgbm.predict_proba(X_val)[:, 1]
accuracy_val = sklearn.metrics.accuracy_score(y_val, y_pred)
auc_val = sklearn.metrics.roc_auc_score(y_val, y_pred_p)
print('Validation accuracy: {:.4f}, AUC: {:.4f}'.format(accuracy_val, auc_val))

# =========== training roc ===========

def lgbm_cross_val_predict(model, X, y, cv=5, cat_features=categorical_features):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    y_pred = np.zeros(len(y))
    y_pred_p = np.zeros(len(y))

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train = y.iloc[train_index]

        model.fit(X_train, y_train, categorical_feature=cat_features)
        y_pred[test_index] = model.predict(X_test)
        y_pred_p[test_index] = model.predict_proba(X_test)[:,1]

    return y_pred, y_pred_p

lgbm_cls = lgb.LGBMClassifier(random_state=42, **grid_search.best_params_)
y_pred, y_pred_p = lgbm_cross_val_predict(lgbm_cls, X_train, y_train)

accuracy = sklearn.metrics.accuracy_score(y_train, y_pred)
auc = sklearn.metrics.roc_auc_score(y_train, y_pred_p)
print('Train accuracy: {:.4f}, Train AUC: {:.4f}'.format(accuracy, auc))


# ======================================= CatBoost  =======================================

split_type = 'random'
all_data = process_df(all_features)
X_train, y_train, X_val, y_val = load_data(all_data, split_type)

# Train CatBoostClassifier
cat_cls = CatBoostClassifier()
param_grid = {
    'iterations': [20, 30],
    'learning_rate': [0.005, 0.01, 0.05],
    'depth': [3, 6]
}
grid_search = GridSearchCV(cat_cls, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train, cat_features=categorical_features)
print("Grid Search - Best Hyperparameters:", grid_search.best_params_)

best_model = CatBoostClassifier(**grid_search.best_params_)
best_model.fit(X_train, y_train, cat_features=categorical_features)

y_pred = best_model.predict(X_val)
y_pred_p = best_model.predict_proba(X_val)[:, 1]

accuracy_val = sklearn.metrics.accuracy_score(y_val, y_pred)
auc_val = sklearn.metrics.roc_auc_score(y_val, y_pred_p)
print('Iter: {}, Validation accuracy: {:.4f}, AUC: {:.4f}'.format(iter_num, accuracy_val, auc_val))
all_result['catboost'] = [y_val, y_pred_p]
# ======== train AUC

cat_cls = CatBoostClassifier(**grid_search.best_params_)
y_pred, y_pred_p = catboost_cross_val_predict(cat_cls, X_train, y_train)

accuracy = sklearn.metrics.accuracy_score(y_train, y_pred)
auc = sklearn.metrics.roc_auc_score(y_train, y_pred_p)
print('Train accuracy: {:.4f}, Train AUC: {:.4f}'.format(accuracy, auc))

# ======== plot feature importance

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)

# Bar plot of feature importance
shap.summary_plot(shap_values, X_val, plot_type="bar")


if False:
    iter_num = 30
    cat_cls = CatBoostClassifier(iterations=iter_num, learning_rate=0.005)
    cat_cls.fit(X_train, y_train, cat_features=categorical_features)

    # Make predictions and evaluate
    y_pred = cat_cls.predict(X_val)
    y_pred_p = cat_cls.predict_proba(X_val)[:, 1]

    accuracy_val = sklearn.metrics.accuracy_score(y_val, y_pred)
    auc_val = sklearn.metrics.roc_auc_score(y_val, y_pred_p)
    print('Iter: {}, Validation accuracy: {:.4f}, AUC: {:.4f}'.format(iter_num, accuracy_val, auc_val))


# ================= Random Forest =================

def process_df_rf(all_features):
    all_features['PDL1_number'] = all_features['PDL1_number'].replace('missing', np.nan)
    all_features['PDL1_number'] = pd.to_numeric(all_features['PDL1_number'])

    trans = 'log1p'
    # === which transformation
    if trans == 'box-cox':
        from scipy.stats import boxcox
        data_positive = all_features['ORR']/100 + 1  # Adding 1 to ensure all values are positive
        transformed_data, lambda_value = boxcox(data_positive)
        all_features['y'] = ((all_features['ORR']/100 + 1)**lambda_value - 1)/lambda_value
    elif trans == 'log1p':
        all_features['y'] = np.log1p(all_features['ORR']/100 + 1)

    categorical_cols = ['性别', '是否吸烟', 'PDL1_expression', '病理诊断_文本', '临床试验分期', '临床试验分期T']
    numerical_cols = [col for col in all_features.columns if col not in categorical_cols + ['imageName', 'Group', 'set', 'ORR', 'y']]

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

    for col in categorical_cols:
        all_features[col] = all_features[col].astype(str)
        print('col {} is str'.format(col))
    for col in numerical_cols:
        all_features[col] = pd.to_numeric(all_features[col])
        print('col {} is numeric'.format(col))

    return all_features, preprocessor

split_type = 'random'
all_data, preprocessor = process_df_rf(all_features)
X_train, y_train, X_val, y_val = load_data(all_data, split_type)


model = RandomForestClassifier(n_estimators=100, random_state=42)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
y_pred_p = clf.predict_proba(X_val)[:, 1]
print("Accuracy: {:.4f}, AUC: {:.4f}".format(accuracy_score(y_val, y_pred), sklearn.metrics.roc_auc_score(y_val, y_pred_p)))

if split_type == 'random':
    random_result['RandomForest'] = [y_val, y_pred_p]


# =====
# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
for key, value in random_result.items():
    y_val, y_pred = value
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    plt.plot(fpr, tpr, lw=2, label=key, alpha=0.5)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Random Split')
plt.legend(loc='lower right')
plt.show()
