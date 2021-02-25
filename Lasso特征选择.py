import preprocessing
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
y=preprocessing.y
x=pd.DataFrame(preprocessing.x_normalized)
columns=x.columns.values
name=preprocessing.columns
data=preprocessing.data_normalized#x归一化，y正常
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
print('Lasso划分time cost(s):',end-start)

#Lasso迭代
start2=time.time()
def cal_fisher(data,isAscending):
	items=list(range(34))
	num_classes = len(set(data[34]))
	fisher_all = []
	grouped = data.groupby([34], as_index=False)
	n = [len(data[data[34] == k+1]) for k in range(num_classes)]
	for i in items:#遍历所有特征列
		temp=grouped[i].agg({str(i)+'_mean': 'mean',str(i)+'_std': 'std'})#求出特征i在各类别k中的均值u_ik、方差p_ik
		numerator=0
		denominator=0
		u_i=data[i].mean()
		for k in range(num_classes):
			n_k = n[k]
			u_ik = temp.iloc[k, :][str(i) + '_mean']
			p_ik = temp.iloc[k, :][str(i) + '_std']
			numerator += n_k * (u_ik - u_i) ** 2
			denominator += n_k * p_ik ** 2
		fisher_all.append(numerator / denominator)
	fisher_all=pd.DataFrame(fisher_all)
	index=fisher_all.index
	fisher_val=fisher_all.values
	fig, (vax) = plt.subplots(1, figsize=(12, 6))
	vax.plot(index, fisher_val, '.')
	vax.vlines(index, [0], fisher_val)
	vax.set_xlabel('index')
	vax.set_title('fisher')
	plt.show()
	return fisher_all.iloc[:,0].sort_values(ascending=isAscending).index.tolist()
def GSIL(x,y,dataList,K):
	x_combine_label=x[dataList].copy()
	y=pd.DataFrame(y)
	y_new_label=y.copy()
	y_new_label.columns=['label']
	new_filter_features = cal_fisher(data,True)
	dataSet = np.array_split(new_filter_features, K)
	dataSet_best=[]
	for item in dataSet:
		currentSet = []
		currentSet.extend(dataSet_best)
		currentSet.extend(item)
		currentX = x[currentSet]
		dataSet_bool = LARS(currentX, y)
		dataSet_best = currentX.columns[dataSet_bool].tolist()
	return dataSet_best
g=GSIL(x,y,columns,5)
end2=time.time()
print('Lasso迭代得到的特征索引值为:',g)
print('Lasso迭代time cost(s):',end2-start2)

