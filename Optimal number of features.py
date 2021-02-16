import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import pandas as pd
import numpy as np
data=pd.read_excel('反向扫视_delta_fea.xlsx')
x=np.array(data.iloc[:,4:])
y=np.array(data.iloc[:,3])
svc=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False)
rfecv=RFECV(estimator=svc,step=1,cv=StratifiedKFold(10),
            scoring='accuracy')
rfecv.fit(x,y)
print(x.shape)
print('Optimal number of features : %d'%rfecv.n_features_)
plt.figure()
plt.xlabel('Number of features selected')
plt.ylabel('Cross validation score (nb of correct classifications)')
plt.plot(range(1,
               len(rfecv.grid_scores_) + 1),
         rfecv.grid_scores_)
plt.show()
