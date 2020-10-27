"""
@Time ： 2020/10/27 16:20
@Auth ： Yifan Hu
@File ：gbdt_plus_lr.py
@IDE ：PyCharm
"""

from data_loader import SklearnDataLoader
from logis_reg import feature_engi
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error

df_data, df_label = SklearnDataLoader.load_builtin('ml-100k', classification=True)
df_data = feature_engi(df_data)
y_true = df_label.values.ravel()
gbdt = GradientBoostingClassifier(n_estimators=40, random_state=0, subsample=.6, max_depth=4, min_samples_split=5)
gbdt.fit(df_data, y_true)

# feature engineering by gbdt
new_data = gbdt.apply(df_data)
new_data = new_data.reshape(-1, 40)
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(new_data)
new_data = one_hot_enc.transform(new_data).toarray()

X_train, X_test, y_train, y_test = train_test_split(new_data, y_true, test_size=0.3, random_state=0)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

print('f1 score : {}'.format(f1_score(y_test, lr_model.predict(X_test))))
print('mse error: {}'.format(mean_squared_error(y_test, lr_model.predict_proba(X_test)[:,1])))