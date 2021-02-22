#要放在划分测试集和训练集之后（只需要对训练集做过采样）
from imblearn.over_sampling import SMOTE
import preprocessing
from collections import Counter
x=preprocessing.x_normalized
y=preprocessing.y
columns=preprocessing.columns

#直接调包使用
sm=SMOTE(random_state=0,n_jobs=-1)
x_smo,y_smo=sm.fit_resample(x,y)
print(Counter(y),Counter(y_smo))


#SMOTE改进
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import copy
class SMOTE():
	"""
	K_neighbors：查找目标对象的几个邻近值
	N_need：将数据扩充几倍
	random_state：随机因子，使每次随机结果相同
	"""

	def __init__(self,
	             K_neighbors=5,
	             N_need=1,
	             random_state=42):
		self.K_neighbors = K_neighbors
		self.N_need = N_need
		self.random_state = 42

	def div_data(self, x_data, y_label):
		"""
		将数据分为少数样本，多数样本，少数样本对应label值，多数样本对应label值，注意样本数据包含label标签值
		"""
		#         print('y_label',y_label)
		tp = set(y_label)
		#         print('tp',tp)
		# 找出结果值出现次数较少的一类
		tp_less = [a for a in tp if sum(y_label == a) < sum(y_label != a)][0]
		#         print('tp_less',tp_less)
		data_less = x_data.iloc[y_label == tp_less, :]
		#         print('tp',tp)
		#         print('data_less',data_less)
		data_more = x_data.iloc[y_label != tp_less, :]
		tp.remove(tp_less)
		return data_less, data_more, tp_less, list(tp)[0]

	def get_SMOTE_sample(self, x_data, y_label):
		"""
		复制少数样本，并追加到少数样本数据中
		"""
		data_less, data_more, tp_less, tp_more = self.div_data(x_data, y_label)

		n_integ = self.N_need
		data_add = copy.deepcopy(data_less)
		#         print('data_less', data_less.shape)
		if n_integ == 0:
			print('WARNING: PLEASE RE-ENTER N_need')
		else:
			for i in range(1, n_integ):  ##扩充少数类的倍数
				data_less = data_less.append(data_add)
		data_less.reset_index(inplace=True, drop=True)
		#         print('data_out', data_less.shape)
		return data_less, tp_less

	def over_sample(self, x_data, y_label):
		"""
		SMOTE算法简单计算公式：
		new_sample = sample[i] + random*(sample[i]-sample.any.neighbor)
		"""
		sample, tp_less = self.get_SMOTE_sample(x_data, y_label)
		knn = NearestNeighbors(n_neighbors=self.K_neighbors, n_jobs=-1).fit(sample)

		n_atters = x_data.shape[1]
		label_out = copy.deepcopy(y_label)
		new = pd.DataFrame(columns=x_data.columns)
		#         print('new', new)

		for i in range(len(sample)):  # 1. 选择一个正样本
			# 2.选择少数类中最近的K个样本
			k_sample_index = knn.kneighbors(np.array(sample.iloc[i, :]).reshape(1, -1),
			                                n_neighbors=self.K_neighbors + 1,
			                                return_distance=False)
			#             print('k_sample_index',type(k_sample_index),k_sample_index)
			#             print('np.array(sample.iloc[i, :]).reshape(1, -1)', np.array(sample.iloc[i, :]).reshape(1, -1))

			# 计算插值样本
			# 3.随机选取K中的一个样本
			np.random.seed(self.random_state)
			choice_all = k_sample_index.flatten()
			#             print('choice_all',choice_all)
			#             print('choice_all[choice_all != 0]',choice_all != 0)
			choosed = np.random.choice(choice_all[choice_all != 0])
			#             print('choosed',choosed)

			# 4. 在正样本和随机样本之间选出一个点
			diff = sample.iloc[choosed,] - sample.iloc[i,]
			#             print('diff',type(diff), diff)
			gap = np.random.rand(1, n_atters)
			#             print('gap', gap)
			#             print('sample.iloc[i,]', sample.iloc[i,])
			new.loc[i] = [x for x in sample.iloc[i,] + gap.flatten() * diff]
			#             print('new',new)

			label_out = np.r_[label_out, tp_less]  ##给新增加的一行数据添加label标签
		print('new', new)
		new_sample = pd.concat([x_data, new])
		new_sample.reset_index(inplace=True, drop=True)
		return new_sample, label_out
if __name__=='__main__':
	x=pd.DataFrame(data=x,columns=columns)
	y=y
	smt = SMOTE()
	x_new, y_new = smt.over_sample(x,y)
	print(Counter(y_new))