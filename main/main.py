import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from math import sqrt
import time
import os

# load data
df = pd.read_csv("data.txt", names=[0, 1, 2, 3], delimiter="\t")

# load data used
data = np.array(df[[1, 2]])
data_with_id = np.array(df[[0, 1, 2]])

# clustering
cluster_num = 12
kmeans_num = 5
agglomerative_num = 5
step = 1
labels = []
sc = []
last_label = 1

print("begin clustering")
for i in range(0, agglomerative_num):
    print(i)
    clf = AgglomerativeClustering(n_clusters=cluster_num)
    clf.fit(data)
    label = clf.labels_ + last_label
    last_label += cluster_num
    # r = metrics.silhouette_score(data, label, metric='euclidean')
    # sc.append(r)
    r = metrics.silhouette_samples(data, label, metric='euclidean')
    sc.extend(r)
    labels.extend(label)
    cluster_num += step

cluster_num = 12
for i in range(0, kmeans_num):
    print(i)
    clf = KMeans(n_clusters=cluster_num, random_state=i)
    clf.fit(data)
    label = clf.labels_ + last_label
    last_label += cluster_num
    # r = metrics.silhouette_score(data, label, metric='euclidean')
    # sc.append(r)
    r = metrics.silhouette_samples(data, label, metric='euclidean')
    sc.extend(r)
    labels.extend(label)
    cluster_num += step

print("clustering over")
data_with_id = np.tile(data_with_id, (kmeans_num + agglomerative_num, 1))

# all data with id and cluster
df = pd.DataFrame(data_with_id, columns=['id', 0, 1])
df['cluster'] = labels
df['sc'] = sc
df.to_csv("all_data.txt", index=False, sep='\t')

# calculate sc for every cluster
cluster_data = df[['cluster', 'sc']].groupby(['cluster'])['sc'].mean().reset_index()
cluster_data['sc'] = list(map(int, (cluster_data['sc'] * 1000000).tolist()))
cluster_data.to_csv("cluster_data.txt", index=False, sep='\t')


# function of get union, difference, intersection of n-dimension list
def get_union(a, b):
    result = []
    for item in a:
        if item not in result:
            result.append(item)
    for item in b:
        if item not in result:
            result.append(item)

    return result


def get_difference(a, b):
    result = []
    for item in a:
        if item not in b:
            result.append(item)
    return result


def get_intersection(a, b):
    result = []
    for item in a:
        if item in b:
            result.append(item)
    return result


def get_avg(l):
    sum = 0
    for i in l:
        sum += i
    return sum / len(l)


# get weight between clusters
def get_weight(a, b):
    coincide_rate = len(get_intersection(a, b)) / len(get_union(a, b))
    if coincide_rate >= 0.2:
        return 0

    alpha = 1 - coincide_rate
    print(alpha)
    X = get_difference(a, get_intersection(a, b))
    Y = get_difference(b, get_intersection(a, b))
    beta = 0.6

    if (len(X) == 0) | (len(Y) == 0):
        t = 0
    else:
        distance = []
        for item1 in X:
            x1 = item1[0]
            y1 = item1[1]
            for item2 in Y:
                x2 = item2[0]
                y2 = item2[1]
                distance.append(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        t = 1 - min(distance) / get_avg(distance)

    return beta * alpha + (1 - beta) * t


# start get weight between clusters
time_start = time.time()
print("start get weight between clusters")
df2 = pd.DataFrame(columns=["i", "j", "weight"])
for i in range(1, last_label - 1):
    a = np.array(df[df.cluster == i][[0, 1]]).tolist()
    for j in range(i + 1, last_label):
        b = np.array(df[df.cluster == j][[0, 1]]).tolist()
        weight = get_weight(a, b)
        weight = int(weight * 1000000 * 0.1)
        df2.loc[len(df2)] = [i, j, weight]
        print([i, j, weight])

time_end = time.time()
print("over")
print('time cost', time_end - time_start, 's')

df2.to_csv("clusters_data.txt", index=False, sep='\t')

# generate graph
cluster_data = pd.read_csv("cluster_data.txt", sep='\t')
clusters_data = pd.read_csv("clusters_data.txt", sep='\t')

count = len(clusters_data)
for i in range(0, len(clusters_data)):
    item = clusters_data.loc[i]
    if item.weight == 0:
        count -= 1

# to graph file
f = open("graph.txt", "w")
f.write("p" + " " + "edge" + " " + str(len(cluster_data)) + " " + str(count) + "\n")
for i in range(0, len(cluster_data)):
    item = cluster_data.loc[i]
    f.write("v" + " " + str(item.cluster) + " " + str(item.sc) + "\n")

for i in range(0, len(clusters_data)):
    item = clusters_data.loc[i]
    if item.weight == 0:
        continue
    f.write("e" + " " + str(item.i) + " " + str(item.j) + " " + str(item.weight) + "\n")

f.close()

# recall mwc.exe
os.system('MWC.exe graph.txt 7 1200')

# final result
# load all data
cluster_data = pd.read_csv("all_data.txt", sep='\t')
cluster_data = cluster_data.drop(['sc'], axis=1)
file = open("best_found_clique.txt", "r")

# load final clusters in best_found_clique.txt
final_clusters = file.readlines()[-1]
final_clusters = final_clusters.strip().split()
final_clusters = list(map(int, final_clusters))

result = pd.DataFrame()
center = pd.DataFrame(columns=['cluster', 'x', 'y'])

for i in final_clusters:
    # get data in final_clusters and save it to the dataframe result
    temp = cluster_data[cluster_data.cluster == i]
    result = result.append(temp)

    # get cluster's center and save it to the dataframe center
    x = temp['0'].mean()
    y = temp['1'].mean()
    center.loc[len(center)] = [i, x, y]

result = result.reset_index(drop=True)
print(center)

# count repeated data points
df = result.groupby(['id', '0', '1']).size().rename('counts').reset_index()
df = df.sort_values(by='counts', ascending=False)
print(df)

# directly save the data points that are not repeated to final result
final_result = pd.DataFrame()
final_result = final_result.append(df[df.counts == 1])
final_result = final_result.drop(['counts'], axis=1)
final_result = pd.merge(final_result, result[['id', 'cluster']], on='id', how='left')

# deal with the data points that repeat 2 times
df_to_do = df[df.counts == 2].reset_index(drop=True)
for i in range(0, len(df_to_do)):
    id = df_to_do.loc[i].id
    x = df_to_do.loc[i]['0']
    y = df_to_do.loc[i]['1']

    # calculate the distance between data point and the two clusters' center
    clusters = np.array(result[result.id == id]['cluster']).tolist()
    cluster1 = clusters[0]
    cluster2 = clusters[1]
    cluster1_x = center[center.cluster == cluster1]['x']
    cluster1_y = center[center.cluster == cluster1]['y']
    cluster2_x = center[center.cluster == cluster2]['x']
    cluster2_y = center[center.cluster == cluster2]['y']

    dis1 = sqrt((x - cluster1_x) ** 2 + (y - cluster1_y) ** 2)
    dis2 = sqrt((x - cluster2_x) ** 2 + (y - cluster2_y) ** 2)

    # put the data point to the cluster with the smallest distance
    if dis1 > dis2:
        final_result.loc[len(final_result)] = [id, x, y, cluster2]
    else:
        final_result.loc[len(final_result)] = [id, x, y, cluster1]

# now all data points' id in final result
final_result_id = set(np.array(final_result['id']).tolist())

# all data points' id
all_data = cluster_data.loc[0:9999].drop(['cluster'], axis=1)
all_data_id = set(np.array(all_data['id']).tolist())

# left data points' id
left_data_id = list(all_data_id - final_result_id)

print(left_data_id)

for i in left_data_id:
    # get data of left data point's id
    data = all_data[all_data.id == i]
    x = data['0']
    y = data['1']

    dis = 999999
    cluster = 0

    # calculate distance between the data point and all clusters' center
    # put the data point to the cluster with the smallest distance
    for j in range(0, len(center)):
        point = center.loc[j]
        x1 = point['x']
        y1 = point['y']
        t = sqrt((x - x1) ** 2 + (y - y1) ** 2)
        if t < dis:
            dis = t
            cluster = point['cluster']

    final_result.loc[len(final_result)] = [i, x, y, cluster]

# replace the clusters' id by range(1, len(final_clusters))
for i in range(0, len(final_clusters)):
    final_result['cluster'] = final_result['cluster'].replace(final_clusters[i], i)

# save final result
final_result = final_result.sort_values(by='id').reset_index(drop=True)
final_result.to_csv("final.txt", index=False, sep='\t')

final_result = np.array(final_result)

# calculate sc of final result
sc = metrics.silhouette_score(final_result[:, 1:3], final_result[:, 3], metric='euclidean')

print(sc)

