from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, accuracy_score
import optuna
from sklearn.preprocessing import LabelEncoder
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

data = pd.concat([train, test], sort=False)
data: Union[DataFrame, Series] = data.drop(['SK_ID_CURR'], axis=1)

le = LabelEncoder()


def missing_values_summary(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'mis_val_count', 1: 'mis_val_percent'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        'mis_val_percent', ascending=False).round(1)
    print("カラム数：" + str(df.shape[1]) + "\n" + "欠損値のカラム数： " + str(mis_val_table_ren_columns.shape[0]))
    return mis_val_table_ren_columns


# for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE',
#                     'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'ORGANIZATION_TYPE', 'NAME_TYPE_SUITE', 'OCCUPATION_TYPE']:
#     data[bin_feature], uniques = pd.factorize(data[bin_feature])
#
# data = data.drop(['OWN_CAR_AGE'], axis=1)

data['DAYS_EMPLOYED_PERC'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
data['INCOME_CREDIT_PERC'] = data['AMT_INCOME_TOTAL'] / data['AMT_CREDIT']
data['INCOME_PER_PERSON'] = data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']
data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
data['PAYMENT_RATE'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']

data['Credit_Annuity_Ratio'] = data['AMT_CREDIT'] / data['AMT_ANNUITY']
data['Credit_goods_price_Ratio'] = data['AMT_CREDIT'] / data['AMT_GOODS_PRICE']
data['Credit_Down_Payment'] = data['AMT_GOODS_PRICE'] - data['AMT_CREDIT']

for col in data.columns:
    if data[col].dtype == object:
        data[col], uniques = pd.factorize(data[col])

knnDatas = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'Credit_Annuity_Ratio']]
for col in knnDatas.columns:
    knnDatas[col].replace([np.inf, -np.inf], np.nan)
    knnDatas[col] = knnDatas[col].fillna(knnDatas[col].mean())
target1 = data[:len(train)]
target1 = target1['TARGET']
temp1 = knnDatas[:len(train)]
temp2 = knnDatas[len(train):]
knn = KNeighborsRegressor(n_neighbors=500)
knn.fit(temp1, target1)
y_pred = knn.predict(knnDatas)
print(y_pred)
data['knn'] = y_pred

train = data[:len(train)]
test = data[len(train):]
X_train = train.drop('TARGET', axis=1)
X_test = test.drop('TARGET', axis=1)
y_train = train['TARGET']

y_preds = []
models = []
oof_train = np.zeros(len((X_train),))
cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=0)

categorical_features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE',
                    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'ORGANIZATION_TYPE','NAME_TYPE_SUITE', 'OCCUPATION_TYPE']

params = {
    'nthread': 4,
    'n_estimators': 10000,
    'colsample_bytree': 0.9497036,
    'objective': 'binary',
    'subsample': 0.8715623,
    'max_depth': 8,
    'reg_alpha': 0.041545473,
    'reg_lambda': 0.0735294,
    'max_bin': 300,
    'num_leaves': 34,
    ' min_split_gain': 0.0222415,
    'min_child_weight': 39.3259775,
    'learning_rate': 0.005,
    'num_iterations': 2000,
    'feature_fraction': 0.38,
    'bagging_fraction': 0.68,
    'bagging_freq': 5,
    'verbose': -1,
    'task': 'train',
    'boosting_type': 'gbdt',
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.loc[train_index, :]
    X_val = X_train.loc[valid_index, :]
    y_tr = y_train.loc[train_index]
    y_val = y_train.loc[valid_index]

    print(f'fold_id: {fold_id}')
    print(f'y_tr y==1: {sum(y_tr)/len(y_tr)}')
    print(f'y_val y==1: {sum(y_val)/len(y_val)}')

    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,  categorical_feature=categorical_features)

    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=2000, early_stopping_rounds=10)
    lgb.plot_importance(model, figsize=(12, 50))

    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_preds.append(y_pred)
    models.append(model)

plt.show()
# pred = pd.DataFrame(oof_train).to_csv('./submitCsv/submission_lightgbm_skfold.csv', index=False)
scores = [m.best_score['valid_1']['binary_logloss'] for m in models]
score = sum(scores) / len(scores)
print('===CV scores===')
print(scores)
print(score)

y_pred_off = (oof_train > 0.5).astype(int)
accuracy_score(y_train, y_pred_off)
len(y_preds)
var = y_preds[0][:10]
y_sub = sum(y_preds) / len(y_preds)

submission['TARGET'] = y_sub
submission.to_csv('../csv/21thSub.csv', index=False)
