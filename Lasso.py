import preprocessing
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import time
y=preprocessing.y
x=pd.DataFrame(preprocessing.x_normalized)
columns=x.columns.values
name=preprocessing.columns

#Lasso划分法
start=time.time()
def LARS(x,y):
	clf=LassoCV(eps=0.0001,n_alphas=300,cv=10,random_state=0,n_jobs=-1).fit(x,y)
	return clf.coef_!=0
def k_split_lasso(x,y,dataList,K):#将特征集合划分为K份,dataList为特征名
	dataSet=np.array_split(dataList,K)
	first_select_features=[]
	for dataSetItem in dataSet:
		x_train_item=x[dataSetItem].copy()
		selected_item_bool=LARS(x_train_item,y)
		selected_item_feature = x_train_item.columns[selected_item_bool].tolist()
		first_select_features+=selected_item_feature
	if len(first_select_features)>0:
		first_select_data=x[first_select_features]
		second_selected_bool=LARS(first_select_data,y)
		return first_select_data.columns[second_selected_bool].tolist()
	else:
		return []
l1=k_split_lasso(x,y,columns,5)
print('Lasso划分得到的特征索引值为:',l1)
end=time.time()
print('time cost(s):',end-start)