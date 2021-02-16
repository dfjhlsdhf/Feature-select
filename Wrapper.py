import preprocessing
from sklearn.feature_selection import RFE,RFECV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,ShuffleSplit,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
columns=preprocessing.columns
x=preprocessing.x_normalized
y=preprocessing.y

#递归特征消除RFE
'''
使用一个基模型（特征含有权重的预测模型coef_，feature_importances_）来进行多轮训练，每轮训练后，消除若干权值系数的特征，
再基于新的特征集进行下一轮训练
RFECV 通过交叉验证的方式执行RFE，以此来选择最佳数量的特征：对于一个数量为d的feature的集合，他的所有的子集的个数是2的d次方减1(包含空集)。
指定一个外部的学习算法，比如SVM之类的。通过该算法计算所有子集的validation error。选择error最小的那个子集作为所挑选的特征
KFold
'''
s=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False)#kernal='linear'才可以用RFE
rfe_s=RFE(s,n_features_to_select=7,step=1)
rfe_s.fit(x,y)
print('Features sorted by their ranks (for svc):')
feature_impotance=(sorted(zip(map(lambda x: round(x, 4), rfe_s.ranking_), columns),reverse=True)[::-1])
print(feature_impotance )
rf=RandomForestClassifier(random_state=0)
rfe_r=RFE(rf,step=1,n_features_to_select=7)
rfe_r.fit(x,y)
print('feature sorted by their rank (for random forest):')
feature_importance=sorted(zip(map(lambda x:round(x,4),rfe_r.ranking_),columns),reverse=True)[::-1]
print(feature_importance)
min_features_to_select=1
rfecv=RFECV(estimator=rf,step=1,cv=StratifiedKFold(5),scoring='accuracy',
            min_features_to_select=min_features_to_select).fit(x,y)
print("Optimal number of features (rfecv) : %d" % rfecv.n_features_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected (rfecv)")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(min_features_to_select,
               len(rfecv.grid_scores_) + min_features_to_select),
         rfecv.grid_scores_)
plt.show()
#Model based ranking
'''
直接使用要用的机器学习算法，针对每个[单独的]特征和响应变量建立预测模型，运用交叉验证
（基于树的方法比较易于使用，因为他们对非线性关系的建模比较好，并且不需要太多的调试。
但要注意过拟合问题，因此树的深度最好不要太大）
'''
# rf2=RandomForestClassifier(n_estimators=20,max_depth=4,random_state=0)
# scores=[]
# #单独采用每个特征进行建模，并进行交叉验证
# '''
# ShuffleSplit类(属于KFold)用于将样本集合随机“打散”后划分为训练集、测试集
# '''
# for i in range(x.shape[1]):
#   score=cross_val_score(rf2,x[:,i:i+1],y,scoring='accuracy',cv=ShuffleSplit(len(x),5,.2))
#   scores.append((format(np.mean(score),'.3f'),columns[i]))
# print('Features sorted by their scores (for rf):',sorted(scores,reverse=True))
# for i in range(x.shape[1]):
#   score=cross_val_score(s,x[:,i:i+1],y,scoring='accuracy',cv=ShuffleSplit(len(x),5,.2))
#   scores.append((format(np.mean(score),'.3f'),columns[i]))
# print('Features sorted by their scores (for svc):',sorted(scores,reverse=True))
