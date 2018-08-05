import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.cluster import KMeans

cluster_data = pd.read_csv("all_data.txt",sep='\t')
use = cluster_data[(cluster_data.cluster<=145) & (cluster_data.cluster>=127)]

data = np.array(use[['0','1']])
clf = KMeans(n_clusters=19,random_state=9)
clf.fit(data)
label = clf.labels_
print(type(label))
sc = metrics.silhouette_score(data, label, metric='euclidean')

print(sc)
# colors = ['r+','g+','b+','c+','y+','ro','go','bo','co','yo',
# 'r*','g*','b*','c*','y*','r.','g.','b.','c.']

# for i in range(0,len(colors)):
# 	print(i)
# 	cluster = i+127
# 	x = use[use.cluster==cluster]['0']
# 	y = use[use.cluster==cluster]['1']
# 	plt.plot(x,y,colors[i])

# plt.show()

use = np.array(use)
print(use[:,1:2])
print(use[:,3])

sc = metrics.silhouette_score(use[:,1:3], use[:,3], metric='euclidean')

print(sc)