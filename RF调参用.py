import matplotlib.pyplot as plt
import math
import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_auc_score
columns=preprocessing.columns
x=preprocessing.x_normalized
y=preprocessing.y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
rf=RandomForestClassifier(random_state=0,oob_score=True)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
print("(初始)AUC score:",roc_auc_score(y_pred,y_test))
print("(初始)十折交叉验证:",cross_val_score(rf,x,y,cv=10).mean())
#param_test1= {'n_estimators':range(900,1400,10)}
# param_test2={'min_samples_split':range(2,10,1),
#              'min_samples_leaf':range(1,10,1)}
param_test3={'max_depth':range(20,30,1)}
gsearch1=GridSearchCV(estimator=RandomForestClassifier(random_state=0,n_estimators=900,oob_score=True,min_samples_split=2,min_samples_leaf=1),
                      param_grid=param_test3,scoring='roc_auc',cv=10)
#min_samples_split=2,min_samples_leaf=1
gsearch1.fit(x,y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)