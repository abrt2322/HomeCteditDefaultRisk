import pandas as pd
import numpy as np
np.random.seed(0)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import time
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import warnings
warnings.simplefilter('ignore', UserWarning)

import gc
gc.enable()

import time

le = LabelEncoder()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

data = pd.concat([train, test], sort=False)
data = data.drop(['SK_ID_CURR'], axis=1)
data = data.replace('XNA', 'F')

for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
    data[bin_feature], uniques = pd.factorize(data[bin_feature])

# data = pd.read_csv("./House_Price/train.csv")
# target = data['SalePrice']

# カテゴリ変数を取得
cat_features = [
    f for f in data.columns if data[f].dtype == 'object'
]

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

for feature in cat_features:
    le = le.fit(data[feature])
    data[feature] = le.transform(data[feature])
    # # カテゴリ変数を数値に変換
    # data[feature], _ = pd.factorize(data[feature])
    # タイプをcategoryに変換
    data[feature] = data[feature].astype('category')

def get_feature_importances(data, cat_features, shuffle, seed=None):
    # 特徴量を取得
    train_features = [f for f in data if f not in 'TARGET']

    # 必要なら目的変数をシャッフル
    y = data['TARGET'].copy()
    if shuffle:
        y = data['TARGET'].copy().sample(frac=1.0)

    # LightGBMで訓練
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2'},
        'num_leaves': 128,
        'learning_rate': 0.01,
        'num_iterations':100,
        'feature_fraction': 0.38,
        'bagging_fraction': 0.68,
        'bagging_freq': 5,
        'verbose': 0
    }
    clf = lgb.train(params=params, train_set=dtrain, num_boost_round=200, categorical_feature=cat_features)

    # 特徴量の重要度を取得
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance"] = clf.feature_importance()

    return imp_df


null_imp_df = pd.DataFrame()
nb_runs = 80
start = time.time()
for i in range(nb_runs):
    imp_df = get_feature_importances(data=data, cat_features=cat_features, shuffle=True)
    imp_df['run'] = i + 1
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

actual_imp_df = get_feature_importances(data=data, cat_features=cat_features, shuffle=False)
feature_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance'].mean()
    imp_score = np.log(1e-10 + f_act_imps / (1 + np.percentile(f_null_imps, 75)))
    feature_scores.append((_f, imp_score))

scores_df = pd.DataFrame(feature_scores, columns=['feature', 'imp_score'])

sorted_features = scores_df.sort_values(by=['imp_score'], ascending=False).reset_index(drop=True)
new_features = sorted_features.loc[sorted_features.imp_score >= 0.5, 'feature'].values
print(new_features)