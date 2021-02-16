'''
Model-based features selection
用cv防止过拟合
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
import preprocessing
columns=preprocessing.columns
names=np.array(columns)
x=preprocessing.x_normalized
y=preprocessing.y
#Embedded1 基于惩罚项(l1 or l2)的特征选择
'''
SelectFromModel(estimation,threshold=None)基于特征重要性，选择重要性高于threshold的特征
SequentialFeatureSelection(estimator,n_features_to_select=None,direction='forward'/'backward',cv=5,scoring='')
对于SVM和逻辑回归，参数C控制稀疏性：C越小，被选中的特征越少
'''
'''
如果如果特征数远远大于样本数的情况下,使用线性核就可以了.
如果特征数和样本数都很大,例如文档分类,一般使用线性核, LIBLINEAR比LIBSVM速度要快很多.
如果特征数远小于样本数,这种情况一般使用RBF.但是如果一定要用线性核,则选择LIBLINEAR较好,而且使用-s 2选项
'''

#使用LinearSCV penalty='l2'做base model 用SelectFromModel做feature select

from sklearn.feature_selection import SelectFromModel
model1=LinearSVC(C=0.01,penalty='l2',dual=False,random_state=0).fit(x,y)#越小惩罚力度越大
sfm=SelectFromModel(model1).fit(x,y)
print('Features selected by SelectFromModel selection l2:',names[sfm.get_support()])

#使用LinearSCV penalty='l2'做base model 用SFS做feature select

from sklearn.feature_selection import SequentialFeatureSelector
model=LinearSVC(C=0.01,penalty='l2',dual=False,random_state=0).fit(x,y)#C越小惩罚力度越大
names=np.array(columns)
sfs_forward=SequentialFeatureSelector(model,n_features_to_select=7,scoring='accuracy',cv=5).fit(x,y)
sfs_backward=SequentialFeatureSelector(model,n_features_to_select=7,scoring='accuracy',direction='backward',cv=5).fit(x,y)
print('Features selected by forward SFS selection l2:',names[sfs_forward.get_support()])
print('Features selected by backward SFS selection l2:',names[sfs_backward.get_support()])

'''
L1惩罚项降维的原理在于保留多个对目标值具有同等相关性的特征中的一个，所以没选到的特征不代表不重要。
故，可结合L2惩罚项来优化。具体操作为：若一个特征在L1中的权值为1，选择在L2中权值差别不大且在L1中权值为0的特征构成同类集合，
将这一集合中的特征平分L1中的权值，故需要构建一个新的模型：
'''
class LS(LinearSVC):
	def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=0.01,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=0,  max_iter=100,
                 multi_class='ovr', verbose=0):
		self.threshold=threshold
		LinearSVC.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose)
		# 使用同样的参数创建L2逻辑回归
		self.l2 = LinearSVC(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
		                             intercept_scaling=intercept_scaling, class_weight=class_weight,
		                             random_state=random_state, max_iter=max_iter,
		                             multi_class=multi_class, verbose=verbose)

	def fit(self, X, y, sample_weight=None):
		# 训练L1逻辑回归
		super(LS, self).fit(X, y, sample_weight=sample_weight)
		self.coef_old_ = self.coef_.copy()
		# 训练L2逻辑回归
		self.l2.fit(X, y, sample_weight=sample_weight)

		cntOfRow, cntOfCol = self.coef_.shape
		# 权值系数矩阵的行数对应目标值的种类数目
		for i in range(cntOfRow):
			for j in range(cntOfCol):
				coef = self.coef_[i][j]
				# L1逻辑回归的权值系数不为0
				if coef != 0:
					idx = [j]
					# 对应在L2逻辑回归中的权值系数
					coef1 = self.l2.coef_[i][j]
					for k in range(cntOfCol):
						coef2 = self.l2.coef_[i][k]
						# 在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
						if abs(coef1 - coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
							idx.append(k)
					# 计算这一类特征的权值系数均值
					mean = coef / len(idx)
					self.coef_[i][idx] = mean
		return self
#带L1和L2惩罚项的逻辑回归作为基模型的特征选择
#参数threshold为权值系数之差的阈值
model2=SelectFromModel(LS(threshold=0.5, C=0.01)).fit(x,y)
print('Features selected by SelectFromModel selection linearSVC_l1+l2:',names[model2.get_support()])

#Embedded2 基于树模型的特征选择
'''
利用SelectFromModel类结合ETC/RF模型
RF是在一个随机子集内得到最佳分叉属性，而ET是完全随机的得到分叉值，从而实现对决策树进行分叉的（分裂随机）
'''
from sklearn.ensemble import ExtraTreesClassifier
etc=ExtraTreesClassifier(random_state=0)
etc.fit(x,y)
importance=etc.feature_importances_
model4=SelectFromModel(etc).fit(x,y)
print('Features selected by SelectFromModel ETC:',names[model4.get_support()])
etc_forward=SequentialFeatureSelector(etc,n_features_to_select=7,scoring='accuracy',cv=5).fit(x,y)
etc_backward=SequentialFeatureSelector(etc,n_features_to_select=7,scoring='accuracy',direction='backward',cv=5).fit(x,y)
print('Features selected by forward SFS selection ETC:',names[etc_forward.get_support()])
print('Features selected by backward SFS selection ETC:',names[etc_backward.get_support()])


#Random Forest + SFS
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state=0)
rf_forward=SequentialFeatureSelector(rf,n_features_to_select=7,scoring='accuracy',cv=5).fit(x,y)
rf_backward=SequentialFeatureSelector(rf,n_features_to_select=7,scoring='accuracy',direction='backward',cv=5).fit(x,y)
print('Features selected by forward SFS selection rf:',names[rf_forward.get_support()])
print('Features selected by backward SFS selection rf:',names[rf_backward.get_support()])
rf.fit(x,y)
#importance_rf=rf.feature_importances_
#Random Forest + SFM
rf_model=SelectFromModel(rf).fit(x,y)
print('Features selected by SelectFromModel RF:',names[rf_model.get_support()])

#LogisticRegression
#L2-based feature selection
from sklearn.linear_model import LogisticRegression
lr1=LogisticRegression(C=0.1,penalty='l2',dual=False,random_state=0).fit(x,y)
lr=SelectFromModel(lr1).fit(x,y)
print('Features selected by SelectFromModel selection lr-l2:',names[lr.get_support()])
lr_forward=SequentialFeatureSelector(lr1,n_features_to_select=7,scoring='accuracy',cv=5).fit(x,y)
lr_backward=SequentialFeatureSelector(lr1,n_features_to_select=7,scoring='accuracy',direction='backward',cv=5).fit(x,y)
print('Features selected by forward SFS selection lr-l2:',names[lr_forward.get_support()])
print('Features selected by backward SFS selection lr-l2:',names[lr_backward.get_support()])

#l1+l2 for logistic
class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        #权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        #使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight = class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        #训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        #训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)

        cntOfRow, cntOfCol = self.coef_.shape
        #权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                #L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    #对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        #在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    #计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self
SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(x,y)
print('Features selected by SelectFromModel selection LR_l1+l2:',names[model2.get_support()])