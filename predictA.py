from cleanData import *
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error
model_names = ['NeuralNet', 'GBR', 'CatBoost', 'XGB', 'LGB']  # 不同模型的名称列表

df = pd.read_csv(".\\data\\level2\\test_A_feat_dummies.csv", encoding='gb18030')
X = np.array(df.drop(['label'], 1))
X = scale_load(X, 'x')

out = pd.DataFrame()
y = 0
modelindex = 0
for model in model_names:
    model = joblib.load(".\\model_add_feature\\" + model_names[modelindex] + ".m")
    y += model.predict(X)
    modelindex += 1

y = y/len(model_names)

for index in range(len(y)):
    original = y[index]
    if original < 0.5:
        y[index] = 0
    else:
        y[index] = 1

out = pd.DataFrame({'y': y})
out.to_csv(r'./result/{}_A_average.csv'.format(datetime.datetime.now().strftime('%m%d_%H%M')), index=None, header=None)




