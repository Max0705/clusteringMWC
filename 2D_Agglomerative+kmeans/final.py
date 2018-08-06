import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import metrics

cluster_data = pd.read_csv("all_data.txt", sep='\t')
cluster_data = cluster_data.drop(['sc'], axis=1)
file = open("best_found_clique.txt", "r")

final_clusters = file.readlines()[-1]
final_clusters = final_clusters.strip().split()
final_clusters = list(map(int, final_clusters))

result = pd.DataFrame()
center = pd.DataFrame(columns=['cluster', 'x', 'y'])

for i in final_clusters:
    temp = cluster_data[cluster_data.cluster == i]
    result = result.append(temp)

    # get cluster center
    x = temp['0'].mean()
    y = temp['1'].mean()
    center.loc[len(center)] = [i, x, y]

result = result.reset_index(drop=True)
print(center)

df = result.groupby(['id', '0', '1']).size().rename('counts').reset_index()
df = df.sort_values(by='counts', ascending=False)

final_result = pd.DataFrame()
final_result = final_result.append(df[df.counts == 1])
final_result = final_result.drop(['counts'], axis=1)
final_result = pd.merge(final_result, result[['id', 'cluster']], on='id', how='left')

df_to_do = df[df.counts == 2].reset_index(drop=True)
for i in range(0, len(df_to_do)):
    id = df_to_do.loc[i].id
    x = df_to_do.loc[i]['0']
    y = df_to_do.loc[i]['1']

    clusters = np.array(result[result.id == id]['cluster']).tolist()
    cluster1 = clusters[0]
    cluster2 = clusters[1]
    cluster1_x = center[center.cluster == cluster1]['x']
    cluster1_y = center[center.cluster == cluster1]['y']
    cluster2_x = center[center.cluster == cluster2]['x']
    cluster2_y = center[center.cluster == cluster2]['y']

    dis1 = sqrt((x - cluster1_x) ** 2 + (y - cluster1_y) ** 2)
    dis2 = sqrt((x - cluster2_x) ** 2 + (y - cluster2_y) ** 2)
    if dis1 > dis2:
        final_result.loc[len(final_result)] = [id, x, y, cluster2]
    else:
        final_result.loc[len(final_result)] = [id, x, y, cluster1]

final_result_id = set(np.array(final_result['id']).tolist())

all_data = cluster_data.loc[0:9999].drop(['cluster'], axis=1)
all_data_id = set(np.array(all_data['id']).tolist())

left_data_id = list(all_data_id - final_result_id)

print(left_data_id)

for i in left_data_id:
    data = all_data[all_data.id == i]
    x = data['0']
    y = data['1']

    dis = 999999
    cluster = 0
    for j in range(0, len(center)):
        point = center.loc[j]
        x1 = point['x']
        y1 = point['y']
        t = sqrt((x - x1) ** 2 + (y - y1) ** 2)
        if t < dis:
            dis = t
            cluster = point['cluster']
    print(cluster)
    final_result.loc[len(final_result)] = [i, x, y, cluster]

for i in range(0, len(final_clusters)):
    print(final_clusters[i])
    final_result['cluster'] = final_result['cluster'].replace(final_clusters[i], i)

final_result = final_result.sort_values(by='id').reset_index(drop=True)
final_result.to_csv("final.txt", index=False, sep='\t')

# colors = ['r+','g+','b+','c+','y+','ro','go','bo','co','yo',
# 'r*','g*','b*','c*','y*','r.','g.','b.','c.']

# for i in range(0,len(colors)):
# 	print(i)
# 	x = final_result[final_result.cluster==i]['0']
# 	y = final_result[final_result.cluster==i]['1']
# 	plt.plot(x,y,colors[i])

# plt.show()
final_result = np.array(final_result)

sc = metrics.silhouette_score(final_result[:, 1:3], final_result[:, 3], metric='euclidean')

print(sc)
