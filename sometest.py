import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso, ElasticNet  # 批量导入要实现的回归算法
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
from tqdm import *
import datetime
import xgboost as xgb
import lightgbm as lgb

from cleanData import *

import warnings
warnings.filterwarnings("ignore")


def make_feat(train, test, encode_cols):
    train_id = train["id"].values.copy()
    test_id = test["id"].values.copy()
    data = pd.concat([train, test])
    # data = train.append(test)
    data = drop_fill(data)
    data['身高体重比'] = data['孕前体重'] / data['身高']
    data = pd.get_dummies(data, columns=encode_cols)
    train_feat = data[data["id"].isin(train_id)]
    test_feat = data[data["id"].isin(test_id)]
    return train_feat, test_feat


train = pd.read_csv('.\\data\\level2\\train.csv', encoding='gb18030')
test = pd.read_csv('.\\data\\level2\\test_A.csv', encoding='gb18030')
encode_cols = listEncodeCols(train)
train_df, test_df = make_feat(train, test, encode_cols)
train_df.to_csv(".\\data\\level2\\train_feat_dummies.csv", encoding='gb18030', index=None)
test_df.to_csv(".\\data\\level2\\test_A_feat_dummies.csv", encoding='gb18030', index=None)

cat_unique_thresh = 3
cat_feature_inds = []
train_features = []
for c in train_df.columns:
    if c not in delcols and c not in "label":
        train_features.append(c)

for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh:
        cat_feature_inds.append(i)

X_train = train_df[train_features]
X_test = test_df[train_features]
y_train = train_df['label']

print("X train , y train shape: ", X_train.shape, y_train.shape)
print("X test shape", X_test.shape)

num_ensembles = 5
y_pred = 0.0


for i in tqdm(range(num_ensembles)):
    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.03,
        depth=6, l2_leaf_reg=3,
        loss_function='Logloss',
        random_seed=i)
    model.fit(
        X_train, y_train, cat_features=cat_feature_inds)
    y_pred += model.predict(X_test)

y_pred /= num_ensembles

for index in range(len(y_pred)):
    pre = y_pred[index]
    if pre not in [0, 1] and pre < 0.5:
        y_pred[index] = 0
    # else:
    #     y_pred[index] = 1

submission = pd.DataFrame({'label': y_pred})

submission.to_csv(r'./result/sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), header=None,
                  index=False, float_format='%.4f')

