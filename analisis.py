import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

df = pd.read_csv("Mall_Customers.csv")

df["is_male"] = (df["Genre"] == "Male").astype(int)
df.drop(columns=["Genre", 'CustomerID'], inplace=True)

#print(df.columns)
#print(df.isnull().sum())

# #===========================================================
# #elbow method thing
# #looks like k-7 is best

# wcss= []
# for k in range(1,20):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(df)
#     wcss.append(kmeans.inertia_)

# plt.plot(range(1,20), wcss, marker="o", linestyle="--")
# plt.show()

# #===========================================================
# ##shows nice results but I'm not sure what I'm seeing

# pca = PCA(n_components=2)
# X = pca.fit_transform(df)

# kmeans = KMeans(n_clusters=6, random_state=42)
# kmeans.fit(X)
# y_kmeans = kmeans.predict(X)

# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
# plt.title("K-Means Clustering")
# plt.legend()
# plt.show()
 
#===========================================================
#nice 3d stuff, possible to make conclusions, but tricky
#virs 40, spending score labai zemas
# zemiausi 10% spending score turi pagrinde vyrai (13/18)
#annual income stipriai krenta zmoniu virs 60 (pensija)
#kaip is nera zmoniu jaunesniu nei 25, kurie uzdirbtu virs 80k
#taip pat tik virs 40 zmones turi aukstesni spending score (virs 60)

# kmeans = KMeans(n_clusters=7, random_state=42)

# #having cluster info as part of df, is there a way to not do that?
# df['Cluster'] = kmeans.fit_predict(df.drop(columns=["is_male"]))

# # 3D Plot initialisation
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# #Markers for is_male (female(0) = ^, male(1) = v)
# markers = {0: '^', 1: 'v'}

# # Plot each category separately based on is_male
# for gender, marker in markers.items():

#     subset = df[df['is_male'] == gender]
#     ax.scatter(subset['Age'], subset['Annual Income (k$)'], subset['Spending Score (1-100)'],
#                c=subset['Cluster'], cmap='viridis', label=f'{"Male" if gender else "Female"}', marker=marker, s=50)

# ax.set_xlabel('Age')
# ax.set_ylabel('Annual Income')
# ax.set_zlabel('Spending Score')
# ax.set_title('3D Clustering Visualization with Gender Representation')
# ax.legend()
# plt.show()
 
#===========================================================
#graph grid
# 0 ir 6 cluster turi panasius spending habbits, tik 0 yra vyresni

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df)

#having cluster info as part of df, is there a way to not do that?
df['Cluster'] = kmeans.fit_predict(df)

grid = sns.PairGrid(df, hue= "Cluster", palette="Set1") #grid_kws={"alpha": 0.8},
grid.map_diag(sns.histplot, multiple="dodge")
grid.map_offdiag(sns.scatterplot) #, kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids'
grid.add_legend()
plt.show()
 
#===========================================================
# #Hierarchical plotting
# linkage_matrix = linkage(df, method="ward") 
# dendrogram(linkage_matrix)
# # plt.figure(figsize=(10,7))
# plt.title("Hierarchical clustering dendogram")
# plt.xlabel("Sample index")
# plt.ylabel("Distance")

# cut_clusters = 150

# clusters = fcluster(linkage_matrix, cut_clusters, criterion="distance")

# df["clusters"] = clusters

# #print(df)

# plt.axhline(y=cut_clusters, color = "r", linestyle="--", label = f"Cut for {cut_clusters} distance")

# plt.show()