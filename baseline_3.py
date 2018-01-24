import time
import datetime
import gc
from dateutil.parser import parse
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
data_path = './data/'

train = pd.read_csv(data_path + 'train.csv', encoding='gb2312')
test = pd.read_csv(data_path + 'test.csv', encoding='gb2312')


def make_feat(train, test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    data = pd.concat([train, test])
    data['性别'] = data['性别'].map({'男': 1, '女': 0, '??': 1})
    data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2016-10-09')).dt.days
    data.fillna(data.median(axis=0),inplace=True)
    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]
    return train_feat, test_feat


train_df, test_df = make_feat(train, test)
print('Remove missing data fields ...')

missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df.shape[0]

for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % len(exclude_missing))
del num_rows, missing_perc_thresh

gc.collect();

print("Remove features with one unique value !!")
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % len(exclude_unique))

print("Define training features !!")

exclude_other = ['id', '血糖', '乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']

train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
 \
            and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % len(train_features))
print(train_features)
print("Define categorial features !!")

cat_feature_inds = []
cat_unique_thresh = 10

for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh:
        cat_feature_inds.append(i)

print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])


print("Training time !!")

X_train = train_df[train_features]

y_train = train_df['血糖']

print(X_train.shape, y_train.shape)

X_test = test_df[train_features]

print(X_test.shape)

print('##################################################################')

# 数据标准化
ss_x = preprocessing.StandardScaler()
X_train = ss_x.fit_transform(X_train)
X_test = ss_x.transform(X_test)


train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(X_train, y_train, train_size=0.8)
# 多层感知器-回归模型
model_mlp = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(20, 20, 20), random_state=1)
model_mlp.fit(train_x_disorder, train_y_disorder)
mlp_score = model_mlp.score(test_x_disorder, test_y_disorder)
print('sklearn多层感知器-回归模型得分', mlp_score)

model_gbr_disorder = GradientBoostingRegressor()
model_gbr_disorder.fit(train_x_disorder, train_y_disorder)
gbr_score_disorder = model_gbr_disorder.score(test_x_disorder, test_y_disorder)
print('sklearn集成-回归模型得分', gbr_score_disorder)  # 准确率较高 0.853817723868


print('###############################参数网格优选###################################')
model_gbr_GridSearch = GradientBoostingRegressor()
#设置参数池  参考 http://www.cnblogs.com/DjangoBlog/p/6201663.html 
param_grid = {'n_estimators': range(10, 101, 10), 'learning_rate': [0.8,1],\
              'min_samples_leaf': range(5, 101, 5), 'max_depth':range(1, 101,5)}

estimator = GridSearchCV(model_gbr_GridSearch, param_grid)
estimator.fit(train_x_disorder, train_y_disorder.ravel())
print('最优调参：', estimator.best_params_)
# {'learning_rate': 0.1, 'max_depth': 6, 'max_features': 0.5, 'min_samples_leaf': 14, 'n_estimators': 70} 

print('调参后得分', estimator.score(test_x_disorder, test_y_disorder))

###画图###########################################################################
model_gbr_best = GradientBoostingRegressor(learning_rate=0.1, max_depth=6, max_features=0.5, min_samples_leaf=14,
                                           n_estimators=40)
model_gbr_best.fit(train_x_disorder, train_y_disorder.ravel())
# 使用默认参数的模型进行预测
gbr_pridict_disorder = model_gbr_disorder.predict(test_x_disorder)
# 多层感知器
mlp_pridict_disorder = model_mlp.predict(test_x_disorder)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
axes = fig.add_subplot(1, 1, 1)
line3, = axes.plot(range(len(test_y_disorder)), test_y_disorder, 'g', label='real')
line1, = axes.plot(range(len(gbr_pridict_disorder)), gbr_pridict_disorder, 'b--', label='gbr', linewidth=2)
line2, = axes.plot(range(len(mlp_pridict_disorder)), mlp_pridict_disorder, 'r--', label='mlp', linewidth=2)
axes.grid()
fig.tight_layout()
plt.legend(handles=[line1, line2, line3])
# plt.legend(handles=[line1,  line3])
plt.title('sklearn regression model')
plt.show()