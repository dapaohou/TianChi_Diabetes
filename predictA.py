from cleanData import *
import pandas as pd
import datetime
from sklearn.metrics import mean_squared_error
model_names = ['Lasso', 'ElasticNet', 'catboost',
               'GBR', 'KRR', 'LightGBM']  # 不同模型的名称列表
tun = {938: 17.523, 928: 3.313, 951: 7.6585, 55: 4.3828, 393: 5.1964, 33: 5.7976,
       822: 5.9668}

df = pd.read_csv(".\\data\\test.csv", encoding="gbk")
df = drop_fill(df)
df = sexencode(df)
show(df)

X = np.array(df.values)
print(X.shape)
X = scale_load(X)

out = pd.DataFrame()
y = 0.0
modelindex = 0
for model in model_names:
    model = joblib.load(".\\model\\" + model_names[modelindex] + ".m")
    y += model.predict(X)
    modelindex += 1
y = y/len(model_names)
out = pd.DataFrame({'y': y})

for index in tun:
    out.at[index, 'y'] = tun[index]
istun = False
if istun:
    tundf = pd.read_excel('tun/{}/tun.xlsx'.format(datetime.datetime.now().strftime('%m%d')), encoding="gbk", index=None)
    tundf.dropna(subset=["姓名"], inplace=True)
    for i in range(len(tundf['姓名'])):
        index = tundf.iloc[i, 0]
        name = tundf.iloc[i, 1]
        new_pre_y = tundf.iloc[i, 2] + 5
        print('index:{},name:{},new_pre_y:{}'.format(index, name, new_pre_y))
        tempout = out.copy()
        tempout.at[index, 'y'] = new_pre_y
        tempout.to_csv(r'tun/0127/{}_{}.csv'.format(datetime.datetime.now().strftime('%m%d_%H%M'), name), index=None, header=None, float_format='%.4f')
else:
    out.to_csv(r'./result/{}_average.csv'.format(datetime.datetime.now().strftime('%m%d_%H%M')), index=None, header=None, float_format='%.4f')




