import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso, ElasticNet  # 批量导入要实现的回归算法
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb

from cleanData import *
from visualization import showliner,showheatmap

import warnings
warnings.filterwarnings("ignore")

from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['font.family']='sans-serif'


df = pd.read_csv(".\\data\\train.csv", encoding='gbk')
df = drop_fill(df)
df = df[df['血糖'] < 30]
df = sexencode(df)
print(df.shape)

X = np.array(df.drop(['血糖'], 1))
y = np.array(df['血糖'])
X = scale_features(X, 'x')
select_features(X, y, df)

# showliner(df['*天门冬氨酸氨基转换酶'], y, '*天门冬氨酸氨基转换酶', '血糖')
showheatmap(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

estimator = lgb.LGBMRegressor(objective='regression')
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=60,boosting_type='gbdt',
                              learning_rate=0.01, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.7,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
param_grid = {
    'learning_rate': [0.001, 0.05, 0.1, 0.5, 0.8, 1],
    'n_estimators': range(100, 1001, 100)
}

gbm = GridSearchCV(estimator, param_grid)

gbm.fit(X_train, y_train)
model_lgb.fit(X_train, y_train)
print('Best parameters found by grid search are:', gbm.best_params_)
print("MSE:\n", mean_squared_error(gbm.predict(X_test), y_test)*0.5)
print('MSE for original:', mean_squared_error(y_test, model_lgb.predict(X_test))*0.5)
