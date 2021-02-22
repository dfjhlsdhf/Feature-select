import numpy as np
import pandas as pd
# f=pd.read_excel("反向扫视.xlsx")
# da=f.columns.values
# for i in range(0,27):
# 	x1=3+i*4
# 	y1=x1+1
# 	x2=y1+1
# 	y2=x2+1
# 	distance=np.sqrt((f[da[x1]]-f[da[x2]])**2+(f[da[y1]]-f[da[y2]])**2)
# 	f['distance'+str(i)]=distance
# f.to_excel("反向扫视_distance.xlsx", index=None)
def distance(v1,v2):
	return np.sqrt(np.sum((v1-v2)**2))
print(distance(np.array([0,0]),np.array([-0.0100531609136731,0.164874910773397])))
v1=np.array([0,0])
v2=np.array([-0.0100531609136731,0.164874910773397])
dist=np.linalg.norm(v1-v2)
print(dist)