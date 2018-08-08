import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from math import sqrt
import time
import os
from nmi import NMI

# define all parameter
datafile = "iris.data"
dimension = 4
cluster_num_begin = 2
step = 1
agglomerative_num = 5
kmeans_num = 5
coincide_rate_threshold = 0.2
beta = 0.6
weight_alpha = 0.1

# load data
df = pd.read_csv(datafile, header=None)

# true label to test
true_label = np.array(df[dimension])

# data to cluster
data = np.array(df.iloc[:, 0:dimension])
dataid = [i for i in range(0, len(df))]
data_with_id = np.column_stack((dataid, data))


def start_cluster():
    print("begin clustering")
    labels = []
    sc = []
    last_label = 1

    # begin two basic cluster functions
    agglomerative_cluster_num = cluster_num_begin
    for i in range(0, agglomerative_num):
        print(i)
        clf = AgglomerativeClustering(n_clusters=agglomerative_cluster_num)
        clf.fit(data)
        label = clf.labels_ + last_label
        last_label += agglomerative_cluster_num
        # r = metrics.silhouette_score(data, label, metric='euclidean')
        # sc.append(r)
        r = metrics.silhouette_samples(data, label, metric='euclidean')
        sc.extend(r)
        labels.extend(label)
        agglomerative_cluster_num += step

    kmeans_cluster_num = cluster_num_begin
    for i in range(0, kmeans_num):
        print(i)
        clf = KMeans(n_clusters=kmeans_cluster_num, random_state=i)
        clf.fit(data)
        label = clf.labels_ + last_label
        last_label += kmeans_cluster_num
        # r = metrics.silhouette_score(data, label, metric='euclidean')
        # sc.append(r)
        r = metrics.silhouette_samples(data, label, metric='euclidean')
        sc.extend(r)
        labels.extend(label)
        kmeans_cluster_num += step

    print("clustering over")
    data_with_id_all = np.tile(data_with_id, (kmeans_num + agglomerative_num, 1))

    # all data with id and cluster
    all_data = pd.DataFrame(data_with_id_all)
    all_data['cluster'] = labels
    all_data['sc'] = sc
    all_data.to_csv("all_data.txt", index=False)

    # calculate sc for every cluster
    cluster_data = all_data[['cluster', 'sc']].groupby(['cluster'])['sc'].mean().reset_index()
    cluster_data['sc'] = list(map(int, (cluster_data['sc'] * 1000000).tolist()))
    cluster_data.to_csv("cluster_data.txt", index=False)


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


# get average of list
def get_avg(l):
    sumtemp = 0
    for i in l:
        sumtemp += i
    return sumtemp / len(l)


# calculate distance of two list
def get_distance(a, b):
    sumtemp = 0
    for i in range(0, len(a)):
        sumtemp += (a[i] - b[i]) ** 2
    return sqrt(sumtemp)


# get weight between clusters
def get_weight(a, b):
    coincide_rate = len(get_intersection(a, b)) / len(get_union(a, b))
    if coincide_rate >= coincide_rate_threshold:
        return 0

    alpha = 1 - coincide_rate
    print(alpha)
    X = get_difference(a, get_intersection(a, b))
    Y = get_difference(b, get_intersection(a, b))

    if (len(X) == 0) | (len(Y) == 0):
        t = 0
    else:
        distance = []
        for item1 in X:
            for item2 in Y:
                distance.append(get_distance(item1, item2))
        t = 1 - min(distance) / get_avg(distance)

    return beta * alpha + (1 - beta) * t


# start get weight between clusters
def start_calculate_weight():
    time_start = time.time()
    print("start get weight between clusters")
    all_data = pd.read_csv("all_data.txt")
    cluster_data = pd.read_csv("cluster_data.txt")
    df2 = pd.DataFrame(columns=["i", "j", "weight"])
    for i in range(1, len(cluster_data)):
        a = np.array(all_data[all_data.cluster == i].iloc[:, 1:dimension + 1]).tolist()
        for j in range(i + 1, len(cluster_data) + 1):
            b = np.array(all_data[all_data.cluster == j].iloc[:, 1:dimension + 1]).tolist()
            weight = get_weight(a, b)
            weight = int(weight * 1000000 * weight_alpha)
            df2.loc[len(df2)] = [i, j, weight]
            print([i, j, weight])

    time_end = time.time()
    print("over")
    print('time cost', time_end - time_start, 's')

    df2.to_csv("clusters_data.txt", index=False)


# generate graph
def start_graph():
    cluster_data = pd.read_csv("cluster_data.txt")
    clusters_data = pd.read_csv("clusters_data.txt")

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


# final result
def start_final_result():
    # load all data
    cluster_data = pd.read_csv("all_data.txt")
    cluster_data = cluster_data.drop(['sc'], axis=1)
    file = open("best_found_clique.txt", "r")

    # load final clusters in best_found_clique.txt
    final_clusters = file.readlines()[-1]
    final_clusters = final_clusters.strip().split()
    final_clusters = list(map(int, final_clusters))

    result = pd.DataFrame()
    center_column = ['cluster'] + [i for i in range(1, dimension + 1)]
    center = pd.DataFrame(columns=center_column)

    for i in final_clusters:
        # get data in final_clusters and save it to the dataframe result
        temp = cluster_data[cluster_data.cluster == i]
        result = result.append(temp)

        # get cluster's center and save it to the dataframe center
        center_temp = [float(i)]
        for j in range(1, dimension + 1):
            center_temp.append(temp.iloc[:, j].mean())

        center.loc[len(center)] = center_temp

    result = result.reset_index(drop=True)

    # count repeated data points
    column_temp = [str(i) for i in range(0, dimension + 1)]
    count_repeated = result.groupby(column_temp).size().rename('counts').reset_index()
    count_repeated = count_repeated.sort_values(by='counts', ascending=False)

    # directly save the data points that are not repeated to final result
    final_result = pd.DataFrame()
    final_result = final_result.append(count_repeated[count_repeated.counts == 1])
    final_result = final_result.drop(['counts'], axis=1)
    final_result = pd.merge(final_result, result[['0', 'cluster']], on='0', how='left')

    # deal with the data points that repeat more than 1 times
    df_to_do = count_repeated[count_repeated.counts >= 2].reset_index(drop=True)
    for i in range(0, len(df_to_do)):
        dataid = df_to_do.loc[i]['0']
        data_temp = []
        for j in range(1, dimension + 1):
            data_temp.append(df_to_do.loc[i][str(j)])

        # get the data point's cluster and its center
        clusters = np.array(result[result['0'] == dataid]['cluster']).tolist()
        cluster_center = []
        for j in range(0, len(clusters)):
            center_temp = center[center.cluster == clusters[j]].iloc[0].tolist()
            center_temp.pop(0)
            cluster_center.append(center_temp)

        # calculate the distance between the data point and the clusters' center
        dis = []
        for j in range(0, len(clusters)):
            dis.append(get_distance(data_temp, cluster_center[j]))

        # put the data point to the cluster with the smallest distance
        j = dis.index(min(dis))
        final_result.loc[len(final_result)] = [dataid] + data_temp + [clusters[j]]

    # now all data points' id in final result
    final_result_id = set(np.array(final_result['0']).tolist())

    # all data points' id
    all_data = cluster_data.loc[0:9999].drop(['cluster'], axis=1)
    all_data_id = set(np.array(all_data['0']).tolist())

    # left data points' id
    left_data_id = list(all_data_id - final_result_id)

    print(left_data_id)

    for i in left_data_id:
        # get data of left data point's id
        data = all_data[all_data['0'] == i]
        data_temp = []
        for j in range(1, dimension + 1):
            data_temp.append(data.iloc[0][str(j)])

        # calculate distance between the data point and all clusters' center
        # put the data point to the cluster with the smallest distance
        dis = 999999
        cluster = 0
        for j in range(0, len(center)):
            point = center.loc[j].tolist()
            point_cluster = point.pop(0)

            t = get_distance(data_temp, point)
            if t < dis:
                dis = t
                cluster = point_cluster

        final_result.loc[len(final_result)] = [i] + data_temp + [cluster]

    # replace the clusters' id by range(1, len(final_clusters))
    for i in range(0, len(final_clusters)):
        final_result['cluster'] = final_result['cluster'].replace(final_clusters[i], i)

    # save final result
    final_result = final_result.sort_values(by='0').reset_index(drop=True)
    final_result.to_csv("final.txt", index=False, sep='\t')

    final_result = np.array(final_result)

    # calculate sc of final result
    sc = metrics.silhouette_score(final_result[:, 1:dimension+1], final_result[:, dimension+1], metric='euclidean')

    print(sc)
    nmi = NMI(true_label, final_result[:, dimension+1])
    print(nmi)


if __name__ == '__main__':
    start_cluster()
    start_calculate_weight()
    start_graph()
    os.system('MWC.exe graph.txt 7 1200')
    start_final_result()
