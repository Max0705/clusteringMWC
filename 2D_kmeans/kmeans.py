import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from math import sqrt
import time

#load data
df = pd.read_csv("data.txt",names=[0,1,2,3],delimiter="\t")

#load data used
data = np.array(df[[1,2]])
data_with_id = np.array(df[[0,1,2]])

#clustering
cluster_num = 10
random_count = 10
step = 1
labels = []
sc = []
last_label = 1
for i in range(0,random_count):
	print(i)
	clf = KMeans(n_clusters=cluster_num,random_state=i)
	clf.fit(data)
	label = clf.labels_ + last_label
	last_label += cluster_num
	r = metrics.silhouette_score(data, label, metric='euclidean')
	sc.append(r)
	labels.extend(label)
	#sc.extend(r)
	cluster_num += step

print(sc)
print(last_label)
data_with_id = np.tile(data_with_id,(random_count,1))

df = pd.DataFrame(data_with_id,columns=['id',0,1])
df['cluster'] = labels
df['sc'] = sc
df.to_csv("all_data.txt",index=False,sep='\t')
get_intersection(())
#calculate sc for every cluster
cluster_data = df[['cluster','sc']].groupby(['cluster'])['sc'].mean().reset_index()
cluster_data['sc'] = list(map(int,(cluster_data['sc']*1000000).tolist()))
cluster_data.to_csv("cluster_data.txt",index=False,sep='\t')

#weight between clusters
def get_union(a,b):
	result = []
	for item in a:
		if item not in result:
			result.append(item)
	for item in b:
		if item not in result:
			result.append(item)

	return result

def get_difference(a,b):
	result = []
	for item in a:
		if item not in b:
			result.append(item)
	return result

def get_intersection(a,b):
	result = []
	for item in a:
		if item in b:
			result.append(item)
	return result

def get_weight(a,b):
	coincide_rate = len(get_intersection(a,b))/len(get_union(a,b))
	if coincide_rate>=0.05:
		return 0

	alpha = 1-coincide_rate
	print(alpha)
	X = get_difference(a,get_intersection(a,b))
	Y = get_difference(b,get_intersection(a,b))
	beta = 0.1

	if (len(X)==0) | (len(Y)==0):
		t = 0
	else :
		distance = []
		for item1 in X:
			x1 = item1[0]
			y1 = item1[1]
			for item2 in Y:
				x2 = item2[0]
				y2 = item2[1]
				distance.append(sqrt((x1 - x2)**2+(y1 - y2)**2))
		t = 1-min(distance)/get_avg(distance)

	return beta*alpha + (1-beta)*t

def get_avg(l):
	sum = 0
	for i in l:
		sum+=i
	return sum/len(l)

time_start=time.time()
df2 = pd.DataFrame(columns = ["i","j","weight"])
for i in range(1,last_label-1):
	a = np.array(df[df.cluster==i][[0,1]]).tolist()
	for j in range(i+1,last_label):
		b = np.array(df[df.cluster==j][[0,1]]).tolist()
		weight = get_weight(a,b)
		weight = int(weight*1000000)
		df2.loc[len(df2)] = [i,j,weight]
		print([i,j,weight])

time_end=time.time()
print('time cost',time_end-time_start,'s')

df2.to_csv("clusters_data.txt",index=False,sep='\t')