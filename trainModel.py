import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, ElasticNet  # 批量导入要实现的回归算法
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法

from cleanData import *


df = pd.read_csv(".\\data\\train.csv", encoding='gbk')
df = drop_fill(df)
df = sexencode(df)

# show(df)
print(df.shape)

X = np.array(df.drop(['血糖'], 1))
y = np.array(df['血糖'])
X = scale_features(X)
select_features(X, y, df)


# 训练回归模型
n_folds = 6  # 设置交叉检验的次数
model_lasso = LassoCV()
model_tree = DecisionTreeRegressor()  # 建立决策树模型对象
model_etc = ElasticNet()  # 建立弹性网络回归模型对象
model_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)  # 建立支持向量机回归模型对象
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
model_names = ['LassoCV', 'DecisionTree', 'ElasticNet', 'SVR', 'GBR']  # 不同模型的名称列表
model_dic = [model_lasso, model_tree, model_etc, model_svr, model_gbr]  # 不同回归模型对象的集合

isTest = False
viewAll = False
if viewAll:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    cv_score_list = []  # 交叉检验结果列表
    pre_y_list = []  # 各个回归模型预测的y值列表

    for model in model_dic:  # 读出每个回归模型对象
        scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
        cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
        pre_y_list.append(model.fit(X_train, y_train).predict(X_test))  # 将回归训练中得到的预测y存入列表

    # 模型效果指标评估
    model_metrics_name = [mean_squared_error, explained_variance_score, mean_absolute_error, r2_score]  # 回归评估指标对象集
    model_metrics_list = []  # 回归评估指标列表
    for i in range(5):  # 循环每个模型索引
        tmp_list = []  # 每个内循环的临时结果列表
        for m in model_metrics_name:  # 循环每个指标对象
            tmp_score = m(y_test, pre_y_list[i])  # 计算每个回归指标结果
            tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
        model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表

    df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
    df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['mse', 'ev', 'mae',  'r2'])  # 建立回归指标的数据框

    print('cross validation result:')  # 打印输出标题
    print(df1)  # 打印输出交叉检验的数据框
    print(70 * '-')  # 打印分隔线
    print('regression metrics:')  # 打印输出标题
    print(df2)  # 打印输出回归指标的数据框
    print(70 * '-')  # 打印分隔线
elif isTest:
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model_gbr.fit(X_train, y_train)
        MSE = mean_squared_error(model_gbr.predict(X_test), y_test)
        print("MSE:", MSE)
else:
    modelindex = 0
    for model in model_dic:
        model.fit(X, y)
        joblib.dump(model, ".\\model\\" + model_names[modelindex] + ".m")
        modelindex += 1


