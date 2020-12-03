import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

data = pd.concat([train, test], sort=False)
data = data.drop(['SK_ID_CURR'], axis=1)

data = data.replace('XNA', 'F')

null_sum = 0
drop_flag = None
for col in data.columns:
    # 欠損の補間
    null_sum = data[col].isnull().sum()
    if null_sum > 0:
        if data[col].dtype == object:
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].mean())

data = data.drop(['OWN_CAR_AGE'], axis=1)

for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
    data[bin_feature], uniques = pd.factorize(data[bin_feature])

data['DAYS_EMPLOYED_PERC'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
data['INCOME_CREDIT_PERC'] = data['AMT_INCOME_TOTAL'] / data['AMT_CREDIT']
data['INCOME_PER_PERSON'] = data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']
data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
data['PAYMENT_RATE'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']

data['Credit_Annuity_Ratio'] = data['AMT_CREDIT'] / data['AMT_ANNUITY']
data['Credit_goods_price_Ratio'] = data['AMT_CREDIT'] / data['AMT_GOODS_PRICE']
data['Credit_Down_Payment'] = data['AMT_GOODS_PRICE'] - data['AMT_CREDIT']

data = pd.get_dummies(data)

train = data[:len(train)]
test = data[len(train):]
X_train = train.drop('TARGET', axis=1)
X_test = test.drop('TARGET', axis=1)
y_train = train['TARGET']

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
          'learning_rate': 0.01, 'num_leaves': 48, 'num_iteration': 5000, 'verbose': 0,
          'colsample_bytree': .8, 'subsample': .9, 'max_depth': 7, 'reg_alpha': .1, 'reg_lambda': .1,
          'min_split_gain': .01, 'min_child_weight': 1}
model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000,
                  early_stopping_rounds=10)
lgb.plot_importance(model, figsize=(12, 50))
plt.show()

X_train = np.array(X_train, np.float32)
X_valid = np.array(X_valid, np.float32)
y_train = np.array(y_train, np.int32)
y_valid = np.array(y_valid, np.int32)

rfc = RandomForestClassifier(random_state=0).fit(X_train, y_train)

import eli5
from eli5.sklearn import PermutationImportance

# permutation importance
perm = PermutationImportance(rfc, random_state=0).fit(X_valid, y_valid)
eli5.show_weights(perm, feature_names=X_test.columns.tolist())