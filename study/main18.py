import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, accuracy_score
import optuna
from sklearn.preprocessing import LabelEncoder
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

data = pd.concat([train, test], sort=False)
data = data.drop(['SK_ID_CURR'], axis=1)

data = data.replace('XNA', 'F')

for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE',
                        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'ORGANIZATION_TYPE',
                        'NAME_TYPE_SUITE', 'OCCUPATION_TYPE']:
    data[bin_feature], uniques = pd.factorize(data[bin_feature])

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

data['DAYS_EMPLOYED_PERC'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
data['INCOME_CREDIT_PERC'] = data['AMT_INCOME_TOTAL'] / data['AMT_CREDIT']
data['INCOME_PER_PERSON'] = data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']
data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
data['PAYMENT_RATE'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']

data['Credit_Annuity_Ratio'] = data['AMT_CREDIT'] / data['AMT_ANNUITY']
data['Credit_goods_price_Ratio'] = data['AMT_CREDIT'] / data['AMT_GOODS_PRICE']
data['Credit_Down_Payment'] = data['AMT_GOODS_PRICE'] - data['AMT_CREDIT']


train = data[:len(train)]
test = data[len(train):]
X_train = train.drop('TARGET', axis=1)
X_test = test.drop('TARGET', axis=1)
y_train = train['TARGET']

y_preds = []
models = []
oof_train = np.zeros(len((X_train),))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


categorical_features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE',
                        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'ORGANIZATION_TYPE',
                        'NAME_TYPE_SUITE', 'OCCUPATION_TYPE']


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)


def objective(trial):
    params1 = {
        'learning_rate': 0.001,
        'num_iterations': 500,
        'objective': 'binary',
        'random_state': 0,
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'max_bin': trial.suggest_int('max_bin', 2, 500),
        'num_leaves': trial.suggest_int('num_leaves', 5, 128),
        'task': 'train',
        'bagging_fraction': 0.68,
        'bagging_freq': 5,
        'verbose': 0,
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'feature_fraction': trial.suggest_uniform('top_rate', 0.0, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 0, 1e-3),
        'lambda_l1': trial.suggest_int('lambda_l1', 0, 500),
        'lambda_l2': trial.suggest_int('lambda_l2', 0, 500),
        'min_gain_to_split': 0,
        'max_depth': trial.suggest_int('max_depth', 5, 10)
    }


    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train,  categorical_feature=categorical_features)

    model1 = lgb.train(params1, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000,
                       early_stopping_rounds=10)
    y_pred_valid = model1.predict(X_valid, num_iteration=model1.best_iteration)
    score = log_loss(y_valid, y_pred_valid)
    return score


study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=40)

params = {
    'learning_rate': 0.001,
    'num_iteration': 500,
    'task': 'train',
    'boosting_type': 'gbdt',
    'num_iterations': 100,
    'bagging_fraction': 0.68,
    'bagging_freq': 5,
    'verbose': 0,
    'min_data_in_leaf': study.best_params['min_data_in_leaf'],
    'feature_fraction': study.best_params['top_rate'],
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': study.best_params['max_depth'],
    'lambda_l1': study.best_params['lambda_l1'],
    'lambda_l2': study.best_params['lambda_l2'],
    'min_child_weight': study.best_params['min_child_weight'],
    'max_bin': study.best_params['max_bin'],
    'num_leaves': study.best_params['num_leaves'],
}

train = data[:len(train)]
test = data[len(train):]
X_train2 = train.drop('TARGET', axis=1)
X_test2 = test.drop('TARGET', axis=1)
y_train2 = train['TARGET']
for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train2, y_train2)):
    X_tr = X_train2.loc[train_index, :]
    X_val = X_train2.loc[valid_index, :]
    y_tr = y_train2.loc[train_index]
    y_val = y_train2.loc[valid_index]

    print(f'fold_id: {fold_id}')
    print(f'y_tr y==1: {sum(y_tr)/len(y_tr)}')
    print(f'y_val y==1: {sum(y_val)/len(y_val)}')

    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,  categorical_feature=categorical_features)

    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000, early_stopping_rounds=10)
    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test2, num_iteration=model.best_iteration)
    y_preds.append(y_pred)
    models.append(model)

# pred = pd.DataFrame(oof_train).to_csv('./submitCsv/submission_lightgbm_skfold.csv', index=False)
scores = [m.best_score['valid_1']['binary_logloss'] for m in models]
score = sum(scores) / len(scores)
print('===CV scores===')
print(scores)
print(score)

y_pred_off = (oof_train > 0.5).astype(int)
accuracy_score(y_train2, y_pred_off)
len(y_preds)
var = y_preds[0][:10]
y_sub = sum(y_preds) / len(y_preds)

submission['TARGET'] = y_sub
submission.to_csv('../csv/18thSub.csv', index=False)