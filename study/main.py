import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')


# データの偏り（targetとなる値の２値分類の偏り）をみる
# count = pd.value_counts(train['TARGET'], sort=True)
# count.plot(kind='bar', rot=0, figsize=(10, 6))
# plt.xticks(range(2), ["0", "1"])
# plt.show()

def missing_values_summary(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'mis_val_count', 1: 'mis_val_percent'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        'mis_val_percent', ascending=False).round(1)
    print("カラム数：" + str(df.shape[1]) + "\n" + "欠損値のカラム数： " + str(mis_val_table_ren_columns.shape[0]))
    return mis_val_table_ren_columns


data = pd.concat([train, test], sort=False)

null_sum = 0
drop_flag = False
for col in data.columns:
    # 欠損の補間
    null_sum = data[col].isnull().sum()
    if null_sum > 0:
        if null_sum / len(data) >= 0.6:
            drop_flag = True
        else:
            if data[col].dtype == object:
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].mean())
    # 不要列の削除
    if drop_flag:
        data = data.drop(col, axis=1)

# データに関する欠損情報や詳細なデータの型などをみる
print(missing_values_summary(train))
print('-'*40)
print(missing_values_summary(test))
print('\n\n\n')

print(train.info())
print('-'*40)
print(test.info())
print('\n\n\n')

data['FLAG_OWN_CAR'].replace(['N', 'Y'], [0, 1], inplace=True)
data['FLAG_OWN_REALTY'].replace(['N', 'Y'], [0, 1], inplace=True)

# for col in data.columns:
#     if data[col].dtype == object:
#         print(pd.get_dummies(data[col]))

# data = pd.get_dummies(data)
#
# train = data[:len(train)]
# test = data[len(train):]
# X_train = train.drop('TARGET', axis=1)
# X_test = test.drop('TARGET', axis=1)
# y_train = train['TARGET']
#
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0, stratify=y_train)
# lgb_train = lgb.Dataset(X_train, y_train)
# lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
# params = {
#     'objective': 'binary'
# }
# model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000,
#                   early_stopping_rounds=10)
# y_pred = model.predict(X_test, num_iteration=model.best_iteration)
# submission['TARGET'] = y_pred
# submission.to_csv('./1stSub.csv', index=False)
