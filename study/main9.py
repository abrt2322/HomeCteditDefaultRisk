import optuna
from sklearn.metrics import log_loss
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

data = pd.concat([train, test], sort=False)


def missing_values_summary(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'mis_val_count', 1: 'mis_val_percent'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        'mis_val_percent', ascending=False).round(1)
    print("カラム数：" + str(df.shape[1]) + "\n" + "欠損値のカラム数： " + str(mis_val_table_ren_columns.shape[0]))
    return mis_val_table_ren_columns


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
    # 不要列の削除
    if drop_flag:
        data = data.drop(col, axis=1)

for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
    data[bin_feature], uniques = pd.factorize(data[bin_feature])

data = data[data['CODE_GENDER'] != 'XNA']
data = data.drop(['OWN_CAR_AGE'], axis=1)

data['DAYS_EMPLOYED_PERC'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
data['INCOME_CREDIT_PERC'] = data['AMT_INCOME_TOTAL'] / data['AMT_CREDIT']
data['INCOME_PER_PERSON'] = data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']
data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
data['PAYMENT_RATE'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']

data = pd.get_dummies(data)
mms = MinMaxScaler()
data[data.columns] = mms.fit_transform(data[data.columns])

train = data[:len(train)]
test = data[len(train):]
X_train = train.drop('TARGET', axis=1)
X_test = test.drop('TARGET', axis=1)
y_train = train['TARGET']

# クラス1の数を保存
count_train_class_one = y_train.sum()
print('クラス1のサンプル数:{}'.format(count_train_class_one)) #クラス1のサンプル数表示

# クラス0：クラス1=9:1になるまでクラス1を増やす
smote = SMOTE(sampling_strategy=0.1, random_state=100)

# 学習用データに反映
x_train_smote, y_train_smote = smote.fit_sample(X_train, y_train)

print(x_train_smote.shape) #学習用データのサンプル数確認
print("SMOTE後のクラス1のサンプル数:{}".format(y_train_smote.sum())) #クラス1のサンプル数
print(type(y_train_smote))
print(type(x_train_smote))
print(y_train_smote.value_counts())

X_train, X_valid, y_train, y_valid = train_test_split(x_train_smote, y_train_smote, test_size=0.3, random_state=0, stratify=y_train_smote)
#
SEED = 0
LR = 0.1
ITER = 100
def objective(trial):
    params = {}
    params['objective'] = 'binary'
    params['random_state'] = SEED
    params['metric'] = 'binary_logloss'
    params['verbosity'] = -1
    params['boosting_type'] = trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss'])

    # モデル訓練のスピードを上げる
    # params['bagging_freq'] = trial.suggest_int('max_bins', 0, 5)
    params['save_binary'] = False

    # 推測精度を向上させる
    params['learning_rate'] = LR
    params['num_iterations'] = ITER
    params['num_leaves'] = trial.suggest_int('num_leaves', 5, 100)
    params['max_bin'] = trial.suggest_int('max_bin', 2, 256)

    # 過学習対策
    # early stoppingは今回使わない。切り方によって、性能を高く見積もる可能性があるため。
    # データ数が少ないため、早期に切り上げる必要性を感じないため。
    # params['early_stopping_round'] = trial.suggest_int('early_stopping_round', 1, 100)
    params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 1, 100)
    params['feature_fraction'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
    # params['bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0, 1.0)
    params['min_child_weight'] = trial.suggest_int('min_child_weight', 0, 1e-3)
    params['lambda_l1'] = trial.suggest_int('lambda_l1', 0, 500)
    params['lambda_l2'] = trial.suggest_int('lambda_l2', 0, 500)
    params['min_gain_to_split'] = 0
    params['max_depth'] = trial.suggest_int('max_depth', 5, 10)

    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
        params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
    if params['boosting_type'] == 'goss':
        params['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
        params['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - params['top_rate'])

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000)
    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    score = log_loss(y_valid, y_pred_valid)
    return score


study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=100)
print('Params: ')
print(study.best_params)

params = {
    'min_data_in_leaf': study.best_params['min_data_in_leaf'],
    'feature_fraction': study.best_params['top_rate'],
    'boosting_type': study.best_params['boosting'],
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.1,
    'num_iteration': 100,
    'max_depth': study.best_params['max_depth'],
    'lambda_l1': study.best_params['lambda_l1'],
    'lambda_l2': study.best_params['lambda_l2'],
    'min_child_weight': study.best_params['min_child_weight'],
    'max_bin': study.best_params['max_bin'],
    'num_leaves': study.best_params['num_leaves'],
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000,
                  early_stopping_rounds=10)

lgb.plot_importance(model, figsize=(12, 50))
plt.show()
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

submission['TARGET'] = y_pred
submission.to_csv('../csv/9thSub.csv', index=False)
