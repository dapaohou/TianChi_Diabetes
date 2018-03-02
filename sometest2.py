from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier  # 集成算法
from catboost import CatBoostClassifier
import datetime
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import f1_score, fbeta_score,fowlkes_mallows_score  # 批量导入指标算法


from cleanData import *

import warnings
warnings.filterwarnings("ignore")


train_df = pd.read_csv(".\\data\\level2\\train_feat_dummies.csv", encoding='gb18030')
test_df = pd.read_csv(".\\data\\level2\\test_A_feat_dummies.csv", encoding='gb18030')


# train_features = []
# for c in train_df.columns:
#     if c not in delcols and c not in "label":
#         train_features.append(c)
#
# X_train = train_df[train_features]
# X_test = test_df[train_features]
# y_train = train_df['label']
#
# print("X train , y train shape: ", X_train.shape, y_train.shape)
# print("X test shape", X_test.shape)
X = np.array(train_df.drop(['label'], 1))
y = np.array(train_df['label'])
X = scale_features(X, 'x')
# select_features(X, y, train_df)

# 训练回归模型
n_folds = 6  # 设置交叉检验的次数
model_knn = KNeighborsClassifier(3)
model_svc_liner = SVC(kernel="linear", C=0.025)
model_svc_gamma = SVC(gamma=2, C=1)
model_Gaussian = GaussianProcessClassifier(1.0 * RBF(1.0))
model_DecisionTree = DecisionTreeClassifier(max_depth=5)
model_RandomForest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
model_MLP = MLPClassifier(alpha=1)
model_AdaBoostClassifier = AdaBoostClassifier()
model_GaussianNB = GaussianNB()
model_QuadraticDis = QuadraticDiscriminantAnalysis()
model_gbr = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1,
                                       max_depth=4, max_features='auto',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='deviance', random_state=5)  # 建立梯度增强回归模型对象
model_catboost = CatBoostClassifier(iterations=1000, learning_rate=0.03,
                                    depth=6, l2_leaf_reg=3,
                                    loss_function='Logloss')
model_xgb = xgb.XGBClassifier(objective="binary:logistic")
model_lgb = lgb.LGBMClassifier(objective='binary')

model_names = ['knn', 'svc_liner', 'svc_rbf', 'Gaussian',
               'DecisionTree', 'RandomForest', 'NeuralNet', 'Adaboost',
               'GaussianNB', 'QuadraticDis', 'GBR', 'CatBoost', 'XGB', 'LGB']  # 不同模型的名称列表
model_dic = [model_knn, model_svc_liner, model_svc_gamma, model_Gaussian,
             model_DecisionTree, model_RandomForest, model_MLP, model_AdaBoostClassifier,
             model_GaussianNB, model_QuadraticDis, model_gbr, model_catboost, model_xgb, model_lgb]  # 不同回归模型对象的集合


# viewAll 为True时查看交叉训练结果，False时生成训练模型
viewAll = False
if viewAll:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    cv_score_list = []  # 交叉检验结果列表
    pre_y_list = []  # 各个回归模型预测的y值列表
    i = 0
    for model in model_dic:  # 读出每个回归模型对象
        scores = cross_val_score(model, X, y, scoring='f1', cv=n_folds)  # 将每个回归模型导入交叉检验模型中做训练检验
        cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
        pre_y = model.fit(X_train, y_train).predict(X_test)
        pre_y_list.append(pre_y)  # 将回归训练中得到的预测y存入列表
        print("model {} train finished!".format(model_names[i]))
        i += 1
    # 模型效果指标评估
    model_metrics_name = [f1_score, fowlkes_mallows_score]  # 回归评估指标对象集
    model_metrics_list = []  # 回归评估指标列表
    for i in range(len(model_dic)):  # 循环每个模型索引
        tmp_list = []  # 每个内循环的临时结果列表
        for m in model_metrics_name:  # 循环每个指标对象
            tmp_score = m(y_test, pre_y_list[i], 'binary')  # 计算每个回归指标结果
            tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
        model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表

    df1 = pd.DataFrame(cv_score_list, index=model_names)  # 建立交叉检验的数据框
    df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['f1', 'fowlkes'])  # 建立回归指标的数据框

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

