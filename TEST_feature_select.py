import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit,StratifiedKFold
from sklearn.feature_selection import SelectFromModel,RFE
from sklearn.linear_model import RandomizedLasso,LogisticRegression
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.metrics import accuracy_score,roc_auc_score,r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from collections import defaultdict
data=pd.read_excel('反向扫视_fea.xlsx')
x=data.iloc[:,4:]
y=data['标签（0：正常；1：异常）']
columns=x.columns
x=(x-x.min())/(x.max()-x.min())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#使用xgboost做特征选择
dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(x_test,label=y_test)
params={'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.3,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':7
	    }
wachlist=[(dtrain,'train'),(dtest,'test')]
model=xgb.train(params,dtrain,num_boost_round=100,evals=(wachlist))
feature_importance=model.get_fscore()#该特征在所有树中被用作分割样本的特征的次数
feature_importance=sorted(feature_importance.items(),key=lambda x:x[1])#x[1]有疑问
print(feature_importance)
plot_importance(model)
pyplot.show()


'''
from numpy import sort
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier,plot_importance
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
data=pd.read_excel('反向扫视_fea.xlsx')
x=data.iloc[:,4:]
y=data.iloc[:,3:4]
x=(x-x.min())/(x.max()-x.min())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
model=XGBClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
predictions=[round(value) for value in y_pred]
accuracy=accuracy_score(y_test,y_pred)
plot_importance(model)
pyplot.show()
print('accuracy:%2.f%%'%(accuracy*100.0))
thresholds=sort(model.feature_importances_)
for thresh in thresholds:
	selections=SelectFromModel(model,threshold=thresh,prefit=True)#带L1和L2惩罚项的逻辑回归作为基模型的特征选择
	select_X_train=selections.transform(x_train)
	selection_model=XGBClassifier()
	selection_model.fit(select_X_train,y_train)
	select_X_test=selections.transform(x_test)
	y_pred=selection_model.predict(select_X_test)
	predictions=[round(value) for value in y_pred]
	accuracy=accuracy_score(y_test,predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))
'''
'''
#随机森林
rf=RandomForestClassifier(random_state=42)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
print(roc_auc_score(y_test,y_pred))
feature_importance=[]
for i in range(len(columns)):
	score = cross_val_score(rf, x_train.values[:, i:i + 1], y_train, scoring="r2",cv=ShuffleSplit(len(x_train), 3, 0.3))
	feature_importance.append((columns[i],round(np.mean(score),3)))
feature_importance.sort(key=lambda x:x[1])
print(feature_importance)
'''

'''
#顶层特征选择算法
'''
#稳定性选择

'''
不可用 rlasso.scores_全是为0.0
rlasso=RandomizedLasso(alpha=0.025)
rlasso.fit(x,y)
print('features sorted by their ranks:')
feature_importance=sorted(zip(map(lambda x:round(x,4),rlasso.scores_),columns))
print(feature_importance)
print(rlasso.scores_)
'''
#RFE

#use SVM as the model
'''
s=svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False)#kernal='linear'才可以用RFE
rfe=RFE(s,n_features_to_select=1)
rfe.fit(x,y)
print('Features sorted by their ranks:')
feature_impotance=(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), columns),reverse=True))
print(feature_impotance)
print(rfe.estimator_)#外部估计函数的相关信息
'''
'''
rf=RandomForestClassifier()
rfe=RFE(rf,n_features_to_select=1,verbose=1)
rfe.fit(x_train,y_train)
print('feature sorted by their rank:')
feature_importance=sorted(zip(map(lambda x:round(x,4),rfe.ranking_),columns),reverse=True)
print(feature_importance)
'''

'''
使用线性模型L1做特征选择
'''
'''
lr=LogisticRegression(penalty='l1',random_state=0,n_jobs=-1)
lr.fit(x_train,y_train)
pred=lr.predict_proba(x_test)[:,1]
print(roc_auc_score(y_test,pred))
#print(lr.coef_)
feature_importance=[(i[0],i[1]) for i in (zip(columns,lr.coef_[0]))]
feature_importance.sort(key=lambda x:np.abs(x[1]))
print(feature_importance)
'''
'''
使用线性模型L2做特征选择
'''
'''
lr2=LogisticRegression(penalty='l2',random_state=0,n_jobs=-1)
lr2.fit(x_train,y_train)
pre=lr2.predict_proba(x_test)[:,1]
pred=lr2.predict(x_test)
#pre=lr2.predict_proba(x_test)
print(roc_auc_score(y_test,pre))
feature_importance=[(i[0],i[1]) for i in (zip(columns,lr2.coef_[0]))]
feature_importance.sort(key=lambda x:np.abs(x[1]))
print(feature_importance)
'''

'''
平均精确率减少
'''



from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
'''
cv可选择的输入为：
-None:使用默认的3折交叉验证
-integer:指定交叉验证的折数
'''
rfecv=RFECV(estimator=rfc, cv=StratifiedKFold(n_splits=3, random_state=0), scoring='neg_mean_squared_error')#estimator估计函数
rfecv.fit(x,y)
print(rfecv.n_features_)
#所选择的属性的个数
#print(rfe.n_features_)
print(rfecv.ranking_)#相应位置上属性的排名,估计最佳的属性被排为1
print(rfecv.estimator_)
print(rfecv.grid_scores_)#交叉验证分数

# from sklearn.cross_validation import cross_val_score, ShuffleSplit
# from sklearn.datasets import load_boston#波士顿房屋价格预测
# from sklearn.ensemble import RandomForestRegressor
# #集成学习ensemble库中的随机森林回归RandomForestRegressor
#
# #Load boston housing dataset as an example
# boston = load_boston()
# X = boston["data"]
# Y = boston["target"]
# names = boston["feature_names"]
# print(boston)
# rf = RandomForestRegressor(n_estimators=20, max_depth=4)
# #20个弱分类器，深度为4
# scores = []
# for i in range(X.shape[1]):#分别让每个特征与响应变量做模型分析并得到误差率
#      score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
#                               cv=ShuffleSplit(len(X), 3, .3))
#      scores.append((round(np.mean(score), 3), names[i]))
# print (sorted(scores, reverse=True))#对每个特征的分数排序