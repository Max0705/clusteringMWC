import random
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import math
import numpy as np
import pandas as pd

d = 3
n = 50000
Pnoise = 0.02
# Pnoise=0
Prestart = 10/(n*(1-Pnoise))
# Prestart = 0
Rshift = 5000*d
Creset = 5000
Rvincintity = 200
result = []

#create random direction and shift the center
def random_direction_shift(center):
	direction = []
	distance = 0
	for i in range(0,d):
		item = random.uniform(-1,1)
		distance += item*item
		direction.append(item)

	distance = math.sqrt(distance)
	for i in range(0,d):
		center[i] = center[i]+Rshift*direction[i]/distance

	return center

#decide if restart according to probability
def restart(p):
	x = random.uniform(0,1)
	if x<=p:
		return 1
	else:
		return 0

#create random data with center and r
def random_direction_with_Rvincintity(center):
	data = []
	direction = []
	distance = 0
	for i in range(0,d):
		item = random.uniform(-1,1)
		distance += item*item
		direction.append(item)

	distance = math.sqrt(distance)
	R = random.uniform(-Rvincintity,Rvincintity)
	for i in range(0,d):
		data.append(center[i]+R*direction[i]/distance)

	return data

#initialize center 
center = []
for i in range(0,d):
	center.append(random.uniform(-500,500))

#create data
margin_max = 0
margin_min = 0
count = 0
break_count = n*(1-Pnoise)
cluster = 0
clusters = []
while 1:
	#restart and shift center
	center = random_direction_shift(center)
	cluster += 1

	for m in range(0,Creset):
		#restart probability
		if restart(Prestart):
			break

		data = random_direction_with_Rvincintity(center)
		center = data
		if margin_max<max(data):
			margin_max=max(data)
		if margin_min>min(data):
			margin_min=min(data)

		result.append(data)
		count += 1
		clusters.append(cluster)

		if count==break_count:
			break

	if count==break_count:
		break

#create noise data
noise_data_count = int(n*Pnoise)
for m in range(0,noise_data_count):
	noise_data = []
	for i in range(0,d):
		noise_data.append(random.uniform(margin_min,margin_max))
	result.append(noise_data)
	clusters.append(-1)

df = pd.DataFrame(result)
df[d] = clusters
df.to_csv("data.txt",header=False,sep='\t')

print(df.shape)
print(cluster)
#dimensionality reduction
result = np.array(result)
pca = PCA(n_components=2)
result = pca.fit_transform(result)

#visualization
plt.plot(result[:,0],result[:,1],'b+')
#plt.plot(result[0:100,0],result[0:100,1],'r+')
plt.show()