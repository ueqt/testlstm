from math import sqrt
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

# load dataset
def parser(x):
    return datetime.strptime(x, '%Y/%m')
series = read_csv('datas/shampoo-sales.csv', header=0, parse_dates=[0],
                  index_col=0, squeeze=True, date_parser=parser)
print(series.head())

series.plot()
pyplot.show()

# 分成训练和测试集合
X = series.values
train, test = X[0:-12], X[-12:]

'''''
步进验证模型:
其实相当于已经用train训练好了模型
之后每一次添加一个测试数据进来
1、训练模型
2、预测一次，并保存预测结构，用于之后的验证
3、加入的测试数据作为下一次迭代的训练数据
'''
history = [x for x in train]
# print(history)
predictions = list()
for i, _ in enumerate(test):
    predictions.append(history[-1]) # history[-1],就是执行预测
    history.append(test[i]) # 将新的测试数据加入模型

# 预测效果评估
# root mean squared error (RMSE)
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE:%.3f' % rmse)

# 画出预测+观测值
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()
