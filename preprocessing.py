'''
MinMaxScaler：是为了消除不同数据之间的量纲，方便数据比较和共同处理，将数据变为（0，1）之间的小数
	   主要算法：min-max归一化——y=(x-min)/(max-min)

标准化StandardScaler：是为了方便数据的下一步处理而进行的数据缩放等变化，并不是为了方便与其他数据一同处理或比较

Normalizer:将每个样本缩放到单位范数（每个样本的范数为1），如果后面要使用如二次型（点积）或者其它核方法计算两个样本之间的相似性这个方法会很有用
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer,MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
data=pd.read_excel('反向扫视_fea.xlsx')
x=data.iloc[:,4:]
columns=x.columns
y=data['标签（0：正常；1：异常）']
y=np.array(y)
#归一化(?
x_normalized=Normalizer().fit_transform(x)
# x_nor=MinMaxScaler().fit_transform(x)
# x_nor=pd.DataFrame(x_nor)
# print(x_nor.head())

#Filter1.去掉取值变化小的特征————方差选择法
'''
#可以用于数据预处理
def get_selected_features():
	variance = VarianceThreshold(threshold=(.98 * (1 - .98))).fit_transform(x)
	print('variance shape:',variance.shape)
	indices=np.argsort(variance)[::-1]
	features=list(x.columns.values[indices[0]])
	print('selected features are:',features)
	return features
var=get_selected_features()
'''