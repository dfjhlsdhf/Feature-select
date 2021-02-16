import pandas as pd
import numpy as np
data1=pd.read_excel('注视_去空_results.xlsx')
data=data1.iloc[:,4:]
dt=data.iloc[:,[i%2==0 for i in range(len(data.columns))]]#x
dt1=data.iloc[:,[i%2==1 for i in range(len(data.columns))]]#y
data1['delta_x_max']=dt.max(axis=1)
data1['delta_x_min']=dt.min(axis=1)
data1['delta_y_max']=dt1.max(axis=1)
data1['delta_y_min']=dt1.min(axis=1)
data1['delta_x_mean']=dt.mean(axis=1)
data1['delta_y_mean']=dt1.mean(axis=1)
data1['delta_x_var']=dt.var(axis=1)#方差
data1['delta_y_var']=dt1.var(axis=1)
data1['delta_x_std']=dt.std(axis=1)#标准差
data1['delta_y_std']=dt1.std(axis=1)
data1['delta_x_median']=dt.median(axis=1)#中位数
data1['delta_y_median']=dt1.median(axis=1)
x_lower=np.quantile(dt,0.25,interpolation='lower',axis=1)
x_higher=np.quantile(dt,0.75,interpolation='higher',axis=1)
data1['delta_x_IQR']=x_higher-x_lower#四分位距
y_lower=np.quantile(dt1,0.25,interpolation='lower',axis=1)
y_higher=np.quantile(dt1,0.75,interpolation='higher',axis=1)
data1['delta_y_IQR']=y_higher-y_lower
data1.to_excel('注视_去空_results_2.xlsx',index=None)