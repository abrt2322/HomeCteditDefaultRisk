import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
import optuna
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

data = pd.concat([train, test], sort=False)
data = data.drop(['SK_ID_CURR'], axis=1)

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


data = data.replace('XNA', 'F')

for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE',
                    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'ORGANIZATION_TYPE']:
    data[bin_feature], uniques = pd.factorize(data[bin_feature])

null_sum = 0
drop_flag = None
for col in data.columns:
    # 欠損の補間
    null_sum = data[col].isnull().sum()
    if null_sum > 0:
        if data[col].dtype == object:
            data[col] = data[col].fillna(data[col].mode()[0])
            data[col], uniques = pd.factorize(data[col])
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

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)


def objective(trial):
    params1 = {
        'objective': 'binary',
        'max_bin': trial.suggest_int('max_bin', 2, 500),
        'learning_rate': 0.05,
        'num_leaves': trial.suggest_int('num_leaves', 32, 128)
    }

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    model1 = lgb.train(params1, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000, early_stopping_rounds=10)
    y_pred_valid = model1.predict(X_valid, num_iteration=model1.best_iteration)
    score = log_loss(y_valid, y_pred_valid)
    return score


study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=40)

params = {
    'objective': 'mean_squared_error',
    'max_bin': study.best_params['max_bin'],
    'num_leaves': study.best_params['num_leaves'],
    'learning_rate': 0.05,
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000,
                  early_stopping_rounds=10)
lgb.plot_importance(model, figsize=(12, 50))
plt.show()

y_pred = model.predict(X_test, num_iteration=model.best_iteration)
submission['TARGET'] = y_pred
submission.to_csv('../csv/14thSub.csv', index=False)
