"""
@Time ： 2020/10/23 8:13
@Auth ： Yifan Hu
@File ：logis_reg.py
@IDE ：PyCharm
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LogisticRegression
from data_loader import SklearnDataLoader
from 数据处理.缺失值处理 import df_fillna
from 数据处理.类别变量处理 import df_get_dummies
from 数据处理.归一化处理 import df_get_and_apply_scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score

target_scaler = MinMaxScaler()


def feature_engi(data):
    # do feature_engineering on data
    data = data.drop(['uid', 'iid', 'timestamp', 'zip code',
                      'movie title', 'release date', 'video release date', 'IMDb URL'], axis=1)
    dummy_cols = ['gender', 'occupation']
    data = df_fillna(data)
    df_dummy = df_get_dummies(data[dummy_cols])
    data = data.drop(dummy_cols, axis=1)
    data = pd.concat([data, df_dummy], axis=1)
    data[['age']], _ = df_get_and_apply_scaler(data[['age']])
    return data


def scale_and_transform(target, inverse=False):
    global target_scaler
    if not inverse:
        # scale target to [0,1]
        return target_scaler.fit_transform(target)
    else:
        # inverse target to original range
        return target_scaler.inverse_transform(target)


def classify_and_transform(target, thres=3):
    target[target < thres] = 0
    target[target >= thres] = 1
    return target


if __name__ == '__main__':
    data, target = SklearnDataLoader.load_builtin('ml-100k')
    data = feature_engi(data)
    # target = scale_and_transform(target)
    target = classify_and_transform(target, thres=3)
    target = np.squeeze(target)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)

    # model = ElasticNet()
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # x = None
    # y = model.predict(x)
    # y_real = scale_and_transform(y, reverse=True)
    print('f1 score : {}'.format(f1_score(y_test, model.predict(X_test))))
    print('mse error: {}'.format(mean_squared_error(y_test, model.predict_proba(X_test)[:, 1])))
