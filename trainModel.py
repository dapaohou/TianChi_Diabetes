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

import xgboost as xgb
import lightgbm as lgb

from cleanData import *

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(".\\data\\level2\\train.csv", encoding='gbk')
checknan(df)
df = drop_fill(df)
df = encode(df)
print(df)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
X = scale_features(X, 'x')
select_features(X, y, df)


# 训练回归模型
n_folds = 6  # 设置交叉检验的次数
model_lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
model_forest = RandomForestRegressor()  # 建立决策树模型对象
model_etc = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))  # 建立弹性网络回归模型对象
model_catboost = CatBoostClassifier(
    iterations=1000, learning_rate=0.03,
    depth=6, l2_leaf_reg=3,
    loss_function='RMSE',
    eval_metric='RMSE')
model_gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1,
                                      max_depth=4, max_features='auto',
                                      min_samples_leaf=15, min_samples_split=10,
                                      loss='ls', random_state=5)  # 建立梯度增强回归模型对象
model_krr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.1, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state=7, nthread=-1)
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=60,boosting_type='gbdt',
                              learning_rate=0.01, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.7,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
model_names = ['Lasso', 'RamdomForest', 'ElasticNet', 'catboost',
               'GBR', 'KRR', 'XGBoost', 'LightGBM']  # 不同模型的名称列表
model_dic = [model_lasso, model_forest, model_etc, model_catboost,
             model_gbr, model_krr, model_xgb, model_lgb]  # 不同回归模型对象的集合


# viewAll 为True时查看交叉训练结果，False时生成训练模型
viewAll = False
if viewAll:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    cv_score_list = []  # 交叉检验结果列表
    pre_y_list = []  # 各个回归模型预测的y值列表

    sum = 0.0
    for model in model_dic:  # 读出每个回归模型对象
        scores = -0.5*cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
        cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
        pre_y = model.fit(X_train, y_train).predict(X_test)
        pre_y_list.append(pre_y)  # 将回归训练中得到的预测y存入列表
        sum += pre_y
    average_pre_y = sum / len(model_names)
    # 模型效果指标评估
    model_metrics_name = [mean_squared_error, explained_variance_score, mean_absolute_error, r2_score]  # 回归评估指标对象集
    model_metrics_list = []  # 回归评估指标列表
    for i in range(len(model_dic)):  # 循环每个模型索引
        tmp_list = []  # 每个内循环的临时结果列表
        for m in model_metrics_name:  # 循环每个指标对象
            tmp_score = m(y_test, pre_y_list[i])  # 计算每个回归指标结果
            tmp_list.append(tmp_score*0.5)  # 将结果存入每个内循环的临时结果列表
        model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
    mse = mean_squared_error(average_pre_y, y_test)
    print('average mse:', mse)
    df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
    df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['mse', 'ev', 'mae',  'r2'])  # 建立回归指标的数据框

    print('cross validation result:')  # 打印输出标题
    print(df1)  # 打印输出交叉检验的数据框
    print(70 * '-')  # 打印分隔线
    print('regression metrics:')  # 打印输出标题
    print(df2)  # 打印输出回归指标的数据框
    print(70 * '-')  # 打印分隔线
else:
    modelindex = 0
    for model in model_dic:
        model.fit(X, y)
        joblib.dump(model, ".\\model_add_feature\\" + model_names[modelindex] + ".m")
        modelindex += 1


