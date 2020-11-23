import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sys
import os, random
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.python.keras import optimizers

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')

data = pd.concat([train, test], sort=False)
data = data.drop(['SK_ID_CURR'], axis=1)


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

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)

X_train = np.array(X_train, np.float32)
X_valid = np.array(X_valid, np.float32)
y_train = np.array(y_train, np.int32)
y_valid = np.array(y_valid, np.int32)


def reset_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed)  # random関数のシードを固定
    np.random.seed(seed)  # numpyのシードを固定
    tf.random.set_seed(seed)  # tensorflowのシードを固定


reset_seed(0)

# モデルの構築
model = tf.keras.models.Sequential([
    tf.keras.layers.BatchNormalization(input_shape=(155,)),
    tf.keras.layers.Dense(155, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(155, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(155, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(78, activation='relu'),
    tf.keras.layers.Dense(39, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# モデルのコンパイル
model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

# モデルの学習
history = model.fit(X_train, y_train,
                    batch_size=10,
                    epochs=50,
                    validation_data=(X_valid, y_valid))

print(history.history)
result = pd.DataFrame(history.history)
print(result.head())

result[['loss', 'val_loss']].plot()
result[['accuracy', 'val_accuracy']].plot()
plt.show()

model.save(filepath='wine_model.h5', save_format='h5')

# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
#
# params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc',
#           'learning_rate': 0.01, 'num_leaves': 48, 'num_iteration': 5000, 'verbose': 0,
#           'colsample_bytree': .8, 'subsample': .9, 'max_depth': 7, 'reg_alpha': .1, 'reg_lambda': .1,
#           'min_split_gain': .01, 'min_child_weight': 1}
#
# # model = lgb.train(params, lgb_train, valid_sets=lgb_eval, early_stopping_rounds=150, verbose_eval=200)
# model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000,
#                   early_stopping_rounds=10)
# lgb.plot_importance(model, figsize=(12, 50))
# plt.show()
#
# y_pred = model.predict(X_test, num_iteration=model.best_iteration)
# submission['TARGET'] = y_pred
# submission.to_csv('../csv/6thSub.csv', index=False)