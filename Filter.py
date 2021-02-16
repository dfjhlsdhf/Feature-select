import preprocessing
from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2,f_classif,mutual_info_classif
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from numpy import array
from minepy import MINE
columns=preprocessing.columns
x=pd.DataFrame(preprocessing.x_normalized,columns=columns)
y=pd.DataFrame(preprocessing.y)

#Filter2.单变量特征选择
'''
SelectKBest: 移除得分前 k 名以外的所有特征(取top k)
SelectPercentile: 移除得分在用户指定百分比以后的特征(取top k%

For regression: f_regression, mutual_info_regression,pearson
For classification: chi2, f_classif, mutual_info_classif互信息
'''
#卡方检验chi2
def get_feature_importance():
	model = SelectKBest(chi2, k=7)  # 选择k个最佳特征
	X_new = model.fit_transform(x, y)
	#print("chi2 shape: ", X_new.shape)
	scores = model.scores_
	#print('chi2 scores:', scores)  # 得分越高，特征越重要
	p_values = model.pvalues_
	#print('chi2 p-values', p_values)  # p-values 越小，置信度越高，特征越重要
	# 按重要性排序，选出最重要的 k 个
	indices = np.argsort(scores)[::-1]#argsort函数返回的是数组值从小到大的索引值
	k_best_features = list(x.columns.values[indices[0:7]])
	print('best features are (for chi2): ', k_best_features)
	return k_best_features
ch2=get_feature_importance()
#f_classif
def get_feature_importance_f():#计算提供的样本的ANOVA（方差分析）F值
	f,pval=f_classif(x,preprocessing.y)
	#print('f_classify F值:',f)#f 值越大，预测能力也就越强，相关性就越大，从而基于此可以进行特征选择
	#print('f_classify p_values:',pval)
	indices=np.argsort(f)[::-1]
	best_features=list(x.columns.values[indices[0:7]])
	print('best features are (for f_classif):',best_features)
	return best_features
f_classi=get_feature_importance_f()
#mutual_info_classif互信息
#MIC的统计能力遭到了 一些质疑 ，当零假设不成立时，MIC的统计就会受到影响。在有的数据集上不存在这个问题，但有的数据集上就存在这个问题。
def get_feature_importance_mic():
	mutual=mutual_info_classif(x,preprocessing.y,discrete_features='auto',copy=True,random_state=0)
	#print(mutual)
	indices=np.argsort(mutual)[::-1]
	best_features=list(x.columns.values[indices[0:7]])
	print('best features are (for MIC):',best_features)
	return best_features
mic=get_feature_importance_mic()
#距离相关系数Distance correlation
#distance_correlation_TEST.py