import pandas as pd
import os

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

os.system('MWC.exe graph.txt 7 1200')
