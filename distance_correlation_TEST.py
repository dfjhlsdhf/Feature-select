import preprocessing
import numpy as np
import pandas as pd
x=preprocessing.x_normalized
y=preprocessing.y
col=x.shape[0]
a=np.zeros((col,col))
b=np.zeros((col,col))
A=np.zeros((col,col))
B=np.zeros((col,col))
#双层循环计算出列之间的范数距离
def dist_corr(x,y):
	for j in range(col):
		for k in range(col):
			a[j,k]=np.linalg.norm(x[j]-x[k])
			b[j,k]=np.linalg.norm(y[j]-y[k])
			#print(a,b)
	#中心化处理
	for m in range(col):
		for n in range(col):
			A[m,n]=a[m,n]-a[m].mean()-a[:,n].mean()+a.mean()
			B[m, n] = b[m, n] - b[m].mean() - b[:, n].mean() + b.mean()
	#求协方差
	cov_xy=np.sqrt((1/(col**2))*((np.dot(A,B)).sum()))
	cov_xx=np.sqrt((1/(col**2))*((np.dot(A,A)).sum()))
	cov_yy=np.sqrt((1/(col**2))*((np.dot(B,B)).sum()))
	print('results are: ',cov_xy/np.sqrt(cov_xy*cov_yy))
	return cov_xy/np.sqrt(cov_xy*cov_yy)
calculate=dist_corr(x,y)


from scipy.spatial.distance import pdist,squareform
import numpy as npr
from numba import jit,float32
