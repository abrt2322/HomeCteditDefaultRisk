import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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

data['FLAG_OWN_CAR'].replace(['N', 'Y'], [0, 1], inplace=True)
data['FLAG_OWN_REALTY'].replace(['N', 'Y'], [0, 1], inplace=True)
data = data.drop(['EXT_SOURCE_1', 'OWN_CAR_AGE'], axis=1)

data = pd.get_dummies(data)
mms = MinMaxScaler()
data[data.columns] = mms.fit_transform(data[data.columns])

print(missing_values_summary(data))

train = data[:len(train)]
test = data[len(train):]
X_train = train.drop('TARGET', axis=1)
X_test = test.drop('TARGET', axis=1)
y_train = train['TARGET']

clf = RandomForestClassifier(oob_score=True, n_estimators=100, max_depth=5)
clf.fit(X_train, y_train)
features = pd.DataFrame(clf.feature_importances_, index=X_train.keys()).sort_values(by=0, ascending=False)
features.plot.bar(legend=False)

params = {
    'n_estimators': [100],
    'random_state': [1],
    'n_jobs': [3],
    'min_samples_split': np.arange(8, 12),
    'max_depth': np.arange(2, 6)
}

clf = GridSearchCV(RandomForestClassifier(), params)
clf.fit(X_train, y_train)
clf = clf.best_estimator_
data = []
num_trials = 10

for i in range(num_trials):
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=i)
    clf.fit(X_train, y_train)
    data.append(clf.score(X_valid, y_valid))

plt.scatter(np.arange(num_trials), data)
plt.ylim([0, 1])
plt.show()

