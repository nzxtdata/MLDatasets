import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift

# Load the wine dataset from scikit-learn
wine = load_wine()
# Convert the dataset into a Pandas DataFrame for simpler analysis
# Reference: (numpy.c_ — NumPy v1.19 Manual, 2020)
# Retrieved from 'https://numpy.org/doc/stable/reference/generated/numpy.c_.html'
df = pd.DataFrame(data=np.c_[wine['target'], wine['data']],
                  columns=['class'] + wine['feature_names'])

# Preview the first 10 rows in the dataset
print(df.head(10))
print("")

# Key aspects of the dataset
print("Features:", wine.feature_names)
print("Targets:", wine.target_names)
print("Total number of values:", df['class'].count())
print("Dimensions of the dataset:", df.shape)
print("")
print("Number of values for each class:\n", df.groupby('class').count())
print("")

# Summary statistics
print("Summary Statistics:")
print(df.describe())
print("")

# Check the data types for each column
print(df.dtypes)
print("")

# Check for null values
print(df.isnull().sum())
print("")

# Preparing the dataset to be used for further machine learning
X = df.drop(['class'], axis=1)
print(X.head())
Y = df.iloc[:, :1]
print(Y.head())
#######################################################################################################################
"""
# Method 1 KMEANS without scaler
"""

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(X)

# Determine the cluster labels of new_points: labels
labels = model.predict(X)

# Print cluster labels of new_points
print(labels)

# Assign the columns of new_points: xs and ys
xs = X.iloc[:, 0]
ys = X.iloc[:, 1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5, cmap='rainbow',edgecolors='b')
# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:, 0]
centroids_y = centroids[:, 1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=75)
plt.title('K-Means without Scalar')
plt.show()

#K-means no Scalar adjusted_rand_score
print(metrics.adjusted_rand_score(df['class'], labels))
#K-means no Scalar metrics
print(metrics.homogeneity_completeness_v_measure(df['class'], labels))
######################################################################################################################
ks = range(1, 11)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(X)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster Sum of Squares')
plt.xticks(ks)
plt.show()
######################################################################################################################
"""#K-Means Method 2 With scaler"""

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=3)

# Create pipeline: pipeline
pipeline: Pipeline = make_pipeline(scaler, kmeans)

pipeline.fit(X)

labels2 = pipeline.predict(X)

print(labels)

#K-means with Scalar adjusted_rand_score
print(metrics.adjusted_rand_score(df['class'], labels2))
#K-means with Scalar metrics
print(metrics.homogeneity_completeness_v_measure(df['class'], labels2))

# Assign the columns of new_points: xs and ys
xs = X.iloc[:, 0]
ys = X.iloc[:, 1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5, cmap='rainbow',edgecolors='b')
plt.title('K-Means with Scalar')
plt.show()

scaler = StandardScaler()
X1 = pd.DataFrame(scaler.fit_transform(X))

print(X1)
#######################################################################################################################
"""#Dendrogram method complete"""

# Calculate the linkage: mergings
mergings = linkage(X, method='complete')

# Plot the dendrogram
dendrogram(mergings,
           leaf_rotation=90,
           leaf_font_size=5,
           )
plt.title('Dendrogram Complete Method')
plt.show()
'''#Dendrogram method ward'''
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram ward Method')

plt.show()
#######################################################################################################################
"""# Birch"""

brc = Birch(branching_factor=50, n_clusters=3, threshold=1.5)
brc.fit(X)

labels3 = brc.predict(X)
xs = X.iloc[:, 0]
ys = X.iloc[:, 1]

plt.scatter(x=xs, y=ys, c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title('BIRCH without scalar')
plt.show()

# BIRCH adjusted_rand_score
print(metrics.adjusted_rand_score(df['class'], labels3))
# BIRCH Metrics
print(metrics.homogeneity_completeness_v_measure(df['class'], labels3))
########################################################################################################################
brc = Birch(branching_factor=50, n_clusters=3, threshold=1.5)
brc.fit(X)

# Create scaler: scaler
scaler = StandardScaler()

# Create pipeline: pipeline
pipeline: Pipeline = make_pipeline(scaler, brc)

pipeline.fit(X)

labels1a = pipeline.predict(X)

print(labels1a)

labels3 = brc.predict(X)
xs = X.iloc[:, 0]
ys = X.iloc[:, 1]

plt.scatter(x=xs, y=ys, c=labels1a, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title('BIRCH with scalar')
plt.show()

# BIRCH adjusted_rand_score
print(metrics.adjusted_rand_score(df['class'], labels1a))
# BIRCH Metrics
print(metrics.homogeneity_completeness_v_measure(df['class'], labels1a))
########################################################################################################################
'''# Affinity Propagation clustering algorithm'''

# define model
model = AffinityPropagation(damping=0.7)

# train the model
model.fit(X)

# assign each data point to a cluster
result = model.predict(X)

# Assign the columns of new_points: xs and ys
xs = X.iloc[:, 0]
ys = X.iloc[:, 1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=result, alpha=0.5, cmap='rainbow',edgecolors='b')
plt.title('AffinityPropagation Clustering')
plt.show()

#AffinityPropagation adjusted_rand_score
print(metrics.adjusted_rand_score(df['class'], result))
#AffinityPropagation Metcics
print(metrics.homogeneity_completeness_v_measure(df['class'], result))
#######################################################################################################################
"""# Mean Shift"""

# define the model
mean_model = MeanShift()

# assign each data point to a cluster
mean_result = mean_model.fit_predict(X)

# Assign the columns of new_points: xs and ys
xs = X.iloc[:, 0]
ys = X.iloc[:, 1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=mean_result, alpha=0.5, cmap='rainbow',edgecolors='b')

# show the Mean-Shift plot
plt.title('MeanShift Clustering')
plt.show()
# MeanShift adjusted_rand_score
print(metrics.adjusted_rand_score(df['class'], mean_result))
# MeanShift Metrics
print(metrics.homogeneity_completeness_v_measure(df['class'], mean_result))
