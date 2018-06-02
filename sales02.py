from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat

# load dataset
def parser(x):
    return datetime.strptime(x, '%Y/%m')
series = read_csv('datas/shampoo-sales.csv', header=0, parse_dates=[0],
                  index_col=0, squeeze=True, date_parser=parser)

'''
将数据转换成有监督数据
即包含input output
训练的目的就是找到训练数据input和output的关系
此处的input是t时间步的数据，output为t+1时间步的数据
具体实现就是将整体的时间数据向后滑动一格，和原始数据拼接，就是有监督的数据
'''
def timeseries_to_supervised(data, lag=1):
    # lag表示的是当前的值只与历史lag个时间步长的值有关，也就是用lag个数据预测下一个
    df = DataFrame(data)
    colums = [df.shift(i) for i in range(1, lag+1)] # 原始数据时间窗向后移动lag步长
    colums.append(df) # 拼接数据
    df = concat(colums, axis=1) # 横向拼接重塑数据，格式:input putput
    df.fillna(0, inplace=True) # 由于数据整体向后滑动lag后，前面的lag个数据是Na形式，用0来填充
    return df

X = series.values
supervised = timeseries_to_supervised(X, 1)
print(supervised.head())
