import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os
import random
import tensorflow as tf

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

data = data.replace('XNA', 'F')

for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
    data[bin_feature], uniques = pd.factorize(data[bin_feature])

drop_flag = None
for col in data.columns:
    if data[col].dtype == object:
        data = data.drop(col, axis=1)

data = data.drop(['OWN_CAR_AGE'], axis=1)

data['DAYS_EMPLOYED_PERC'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
data['INCOME_CREDIT_PERC'] = data['AMT_INCOME_TOTAL'] / data['AMT_CREDIT']
data['INCOME_PER_PERSON'] = data['AMT_INCOME_TOTAL'] / data['CNT_FAM_MEMBERS']
data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
data['PAYMENT_RATE'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']

train = data[:len(train)]
test = data[len(train):]
X_train = train.drop('TARGET', axis=1)
X_test = test.drop('TARGET', axis=1)
y_train = train['TARGET']

print(type(X_train.values))
print(np.unique(y_train.values))
print(type(y_train.values))

X_train, X_valid, y_train, y_valid = train_test_split(X_train.values, y_train.values, train_size=0.7, random_state=0,
                                                      stratify=y_train)

print(X_train.shape)
print(X_train.dtype)
print(X_valid.shape)
print(X_valid.dtype)
print('*' * 40)
print(y_train.shape)
print(y_train.dtype)
print(y_valid.shape)
print(y_valid.dtype)
print('\n' * 2)

X_train = np.array(X_train, np.float32)
X_valid = np.array(X_valid, np.float32)
y_train = np.array(y_train, np.int32)
y_valid = np.array(y_valid, np.int32)

print(X_train.shape)
print(X_train.dtype)
print(X_valid.shape)
print(X_valid.dtype)
print('*' * 40)
print(y_train.shape)
print(y_train.dtype)
print(y_valid.shape)
print(y_valid.dtype)


def reset_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed)  # random関数のシードを固定
    np.random.seed(seed)  # numpyのシードを固定
    tf.random.set_seed(seed)  # tensorflowのシードを固定


reset_seed(0)

# # モデルの構築
# model = tf.keras.models.Sequential([
#     tf.keras.layers.BatchNormalization(input_shape=(45,)),
#     tf.keras.layers.Dense(45, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(20, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid'),
# ])
#
# # モデルのコンパイル
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# # モデルの学習
# history = model.fit(X_train, y_train,
#                     batch_size=10,
#                     epochs=10,
#                     validation_data=(X_valid, y_valid))
#
# print(history.history)
# result = pd.DataFrame(history.history)
# print(result.head())
#
# result[['loss', 'val_loss']].plot()
# result[['accuracy', 'val_accuracy']].plot()
# plt.show()
#
# model.save(filepath='model.h5', save_format='h5')

loaded_model = tf.keras.models.load_model('model.h5')
y = loaded_model.predict(X_test.values)
submission['TARGET'] = y
submission.to_csv('../csv/7thSub.csv', index=False)