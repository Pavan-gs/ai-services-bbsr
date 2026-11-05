import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("E:/data/cust.csv")
df[:5]
df1 = df.iloc[:,2:]

# K-means algorithm
# Euclidean distance --> sqrt(x2-x1)2+(y2-y1)2
# Standardise the data

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
df_scaled = scale.fit_transform(df1)
df_scaled[:5]

from sklearn.cluster import KMeans
clust1 = KMeans(n_clusters=3, random_state=2)
clust_num = clust1.fit_predict(df_scaled)

# Elbow plot

wss = []

for i in range(1,11):
    cluster = KMeans(n_clusters=i, random_state=8)
    cluster.fit(df_scaled)
    wss.append(cluster.inertia_)
plt.plot(range(1,11), wss)
plt.title("Elbow method")
plt.xlabel("Num of clusters")
plt.ylabel("Within ss")
plt.show()

from sklearn.cluster import KMeans
clust1 = KMeans(n_clusters=5, random_state=8)
clust_num = clust1.fit_predict(df_scaled)

final_df = df.copy()
final_df['cluster_num'] = clust_num

clust1.inertia_
clust1.labels_

grp = final_df.groupby("cluster_num")
grp
grp.size()

df.columns

avg = grp.apply(np.mean)
insight1=final_df[['Fresh', 'Milk', 'Grocery', 'Frozen','Detergents_Paper', 'Delicassen',
                   'cluster_num']].groupby('cluster_num').mean()

final_df[final_df['cluster_num']==3]

from sklearn.metrics import silhouette_score
sil_score = silhouette_score(df_scaled,clust_num)

for k in range(2,10):
    km = KMeans(n_clusters=k,random_state=8)
    labels = km.fit_predict(df_scaled)
    score = silhouette_score(df_scaled,labels)
    print(f"k={k}, sc = {score}")

# Hierarchical Clustering

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

data = pd.read_csv('E://data/europe11.csv')
data1 = data.iloc[:,1:]
from sklearn import preprocessing
x_scaled = preprocessing.scale(data1)
dist = pdist(x_scaled)

link = linkage(dist, method='complete')

plt.figure(figsize=(10,6))
dendrogram(link, labels=list(data.Country))
plt.xlabel("Countries")
plt.ylabel("distance")
plt.show()

#k-means

clust2 = KMeans(n_clusters=5, random_state=8)
clust_num = clust2.fit_predict(data1)

data['cnum']=clust_num

grp1=data.iloc[:,1:].groupby('cnum')
grp1.size()

centroids = clust2.cluster_centers_
labels = clust2.labels_
colors = ["g.","r.","b.","y.","k."]


# plot the clusters in a 2 dimensional chart

x = pd.DataFrame(data1).iloc[:,[0,1]]
for i in range(len(data1)):
    plt.plot(x.iloc[i,0],x.iloc[i,1], colors[labels[i]], markersize=10)
plt.xlabel("Area")
plt.ylabel("GDP")
plt.show()


# k-means ---> Data is continuous, number of clusters is known, spherical clusters
# DBScan --> Good for outliers, noise, unknown number of clusters
# k-medoids --> Categorical, Robust to outliers
# Hierarchical --> Hierarchy, visualization










